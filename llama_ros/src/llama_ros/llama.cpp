// MIT License
//
// Copyright (c) 2023 Miguel Ángel González Santamarta
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <algorithm>
#include <cassert>
#include <chat.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <llama.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "build-info.h"
#include "llama_utils/chat_utils.hpp"
#include "sampling.h"
#include "speculative.h"

#include "llama_ros/llama.hpp"
#include "llama_ros/metadata_utils.hpp"
#include "llama_utils/logs.hpp"

using namespace llama_ros;

namespace {

/**
 * @brief Install a CPU abort callback only during one decode call.
 *
 * Llama owns this context and normally has no callback. Synchronize before
 * removing the callback so no backend retains a pointer to its stack data.
 */
class ScopedDecodeAbort {
public:
  ScopedDecodeAbort(llama_context *ctx, std::function<bool()> callback)
      : ctx_(ctx), callback_(std::move(callback)) {
    if (callback_) {
      llama_set_abort_callback(
          ctx_,
          [](void *data) {
            return (*static_cast<std::function<bool()> *>(data))();
          },
          &callback_);
    }
  }
  ~ScopedDecodeAbort() {
    if (callback_) {
      llama_synchronize(ctx_);
      llama_set_abort_callback(ctx_, nullptr, nullptr);
    }
  }
  ScopedDecodeAbort(const ScopedDecodeAbort &) = delete;
  ScopedDecodeAbort &operator=(const ScopedDecodeAbort &) = delete;

private:
  llama_context *ctx_;
  std::function<bool()> callback_;
};

/**
 * @brief Snapshot a sequence state into @p data. Returns false when the state
 * cannot be read consistently (e.g. the sequence was cleared concurrently),
 * instead of aborting like common_prompt_checkpoint::update_tgt().
 */
bool save_seq_state(std::vector<uint8_t> &data, llama_context *ctx,
                    llama_seq_id seq_id, llama_state_seq_flags flags) {
  if (ctx == nullptr) {
    return true;
  }

  const size_t size = llama_state_seq_get_size_ext(ctx, seq_id, flags);
  if (size == 0) {
    return false;
  }

  data.resize(size);
  const size_t n =
      llama_state_seq_get_data_ext(ctx, data.data(), size, seq_id, flags);
  if (n != size) {
    data.clear();
    return false;
  }

  return true;
}

/**
 * @brief Restore a sequence state previously captured with save_seq_state().
 */
bool load_seq_state(const std::vector<uint8_t> &data, llama_context *ctx,
                    llama_seq_id seq_id, llama_state_seq_flags flags) {
  if (ctx == nullptr || data.empty()) {
    return true;
  }

  const size_t n = llama_state_seq_set_data_ext(ctx, data.data(), data.size(),
                                                seq_id, flags);
  return n == data.size();
}

/**
 * @brief Escapes a string the way llama.cpp does when stringifying arrays.
 */
std::string gguf_escape_string(const std::string &value) {
  std::string escaped;
  escaped.reserve(value.size());

  for (const char c : value) {
    if (c == '\\' || c == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(c);
  }

  return escaped;
}

/**
 * @brief Stringifies a scalar GGUF value, mirroring llama.cpp's formatting.
 */
std::string gguf_scalar_to_string(gguf_type type, const void *data,
                                  size_t index) {
  switch (type) {
  case GGUF_TYPE_UINT8:
    return std::to_string(static_cast<const uint8_t *>(data)[index]);
  case GGUF_TYPE_INT8:
    return std::to_string(static_cast<const int8_t *>(data)[index]);
  case GGUF_TYPE_UINT16:
    return std::to_string(static_cast<const uint16_t *>(data)[index]);
  case GGUF_TYPE_INT16:
    return std::to_string(static_cast<const int16_t *>(data)[index]);
  case GGUF_TYPE_UINT32:
    return std::to_string(static_cast<const uint32_t *>(data)[index]);
  case GGUF_TYPE_INT32:
    return std::to_string(static_cast<const int32_t *>(data)[index]);
  case GGUF_TYPE_UINT64:
    return std::to_string(static_cast<const uint64_t *>(data)[index]);
  case GGUF_TYPE_INT64:
    return std::to_string(static_cast<const int64_t *>(data)[index]);
  case GGUF_TYPE_FLOAT32:
    return std::to_string(static_cast<const float *>(data)[index]);
  case GGUF_TYPE_FLOAT64:
    return std::to_string(static_cast<const double *>(data)[index]);
  case GGUF_TYPE_BOOL:
    return static_cast<const int8_t *>(data)[index] != 0 ? "true" : "false";
  default:
    return "";
  }
}

/**
 * @brief Stringifies a flat GGUF array like llama.cpp's gguf_kv_to_str().
 */
std::string gguf_array_to_string(const gguf_context *ctx, int64_t key_id) {
  const gguf_type arr_type = gguf_get_arr_type(ctx, key_id);
  const size_t count = gguf_get_arr_n(ctx, key_id);
  const void *data =
      arr_type == GGUF_TYPE_STRING ? nullptr : gguf_get_arr_data(ctx, key_id);

  std::string out = "[";
  for (size_t i = 0; i < count; ++i) {
    if (i > 0) {
      out += ", ";
    }

    if (arr_type == GGUF_TYPE_STRING) {
      out += '"' + gguf_escape_string(gguf_get_arr_str(ctx, key_id, i)) + '"';
    } else if (arr_type == GGUF_TYPE_ARRAY) {
      out += "???";
    } else {
      out += gguf_scalar_to_string(arr_type, data, i);
    }
  }
  out += "]";
  return out;
}

/**
 * @brief Stringifies a scalar GGUF metadata value.
 */
std::string gguf_value_to_string(const gguf_context *ctx, int64_t key_id) {
  const gguf_type type = gguf_get_kv_type(ctx, key_id);

  if (type == GGUF_TYPE_STRING) {
    const char *value = gguf_get_val_str(ctx, key_id);
    return value == nullptr ? "" : value;
  }

  if (type == GGUF_TYPE_ARRAY) {
    return gguf_array_to_string(ctx, key_id);
  }

  return gguf_scalar_to_string(type, gguf_get_val_data(ctx, key_id), 0);
}

} // namespace

Llama::Llama(const common_params &params, std::string system_prompt,
             bool initial_reset)
    : params(params), system_prompt(system_prompt) {

  this->llama_init = common_init_from_params(this->params);

  llama_print_build_info(llama_version());

  // load model
  llama_backend_init();
  llama_numa_init(this->params.numa);

  this->model = this->llama_init->model();
  this->ctx = this->llama_init->context();
  this->lora_adapters = this->params.lora_adapters;

  // raw GGUF metadata for values the model API does not expose (arrays)
  if (!this->params.model.path.empty()) {
    gguf_init_params gguf_params = {};
    gguf_params.no_alloc = true;
    this->gguf_metadata_ =
        gguf_init_from_file(this->params.model.path.c_str(), gguf_params);

    if (this->gguf_metadata_ == nullptr) {
      LLAMA_LOG_WARN("Failed to read raw GGUF metadata from %s",
                     this->params.model.path.c_str());
    }
  }

  if (this->model == NULL) {
    LLAMA_LOG_ERROR("Unable to load model");
    throw std::runtime_error("Unable to load model");
  }

  // Context checkpoints are only useful when partial seq_rm is unsupported or
  // the model uses SWA (mirrors llama.cpp server-context.cpp). Note that
  // common_context_can_seq_rm() clears the context memory, so query it once.
  const auto seq_rm_type = common_context_can_seq_rm(this->ctx);
  this->partial_seq_removal_ = seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_PART;
  if (this->params.n_ctx_checkpoints > 0) {
    this->checkpoints_enabled_ =
        seq_rm_type != COMMON_CONTEXT_SEQ_RM_TYPE_PART ||
        llama_model_n_swa(this->model) > 0;
  }

  // Host-RAM prompt cache (0 disables, < 0 means unlimited)
  if (this->params.cache_ram_mib != 0) {
    this->prompt_cache_ = std::make_unique<PromptCache>(
        this->params.cache_ram_mib, this->params.n_ctx);
  }

  // Slots
  const int32_t n_ctx_slot = this->params.n_ctx / this->params.n_parallel;
  LLAMA_LOG_INFO("slot context size: %d", n_ctx_slot);
  LLAMA_LOG_INFO("n_parallel: %d", this->params.n_parallel);

  for (int i = 0; i < this->params.n_parallel; i++) {
    ServerSlot slot;
    slot.id = i;
    slot.ctx = this->llama_init->context();
    slot.n_ctx = n_ctx_slot;
    slot.n_predict = this->params.n_predict;
    slot.params.n_predict = this->params.n_predict;
    slot.params.sampling = this->params.sampling;
    slot.params.n_keep = this->params.n_keep;
    slot.sampler = nullptr;

    slot.reset();
    this->server_slots.push_back(std::move(slot));
  }

  LLAMA_LOG_INFO("Initializing batch context");

  this->batch = common_batch(this->ctx);

  LLAMA_LOG_INFO("Model loaded successfully");

  // Initialize managers and handlers
  this->slot_manager_ = std::make_unique<SlotManager>(this->server_slots);
  this->task_registry_ = std::make_unique<TaskRegistry>();
  LLAMA_LOG_INFO("Initialized Slot Manager and Task Registry");

  this->embedding_handler_ = std::make_unique<EmbeddingRequestHandler>(this);
  this->decision_handler_ = std::make_unique<DecisionRequestHandler>(this);
  this->rerank_handler_ = std::make_unique<RerankRequestHandler>(this);
  this->completion_handler_ = std::make_unique<CompletionRequestHandler>(this);
  this->chat_completion_handler_ =
      std::make_unique<ChatCompletionRequestHandler>(this);
  LLAMA_LOG_INFO("Initialized Request Handlers");

  // Initialize chat formatter
  this->chat_formatter_ = std::make_unique<llama_utils::ChatFormatter>(
      this->model, this->params.chat_template);

  this->oai_parser_opt = {this->params.use_jinja,
                          this->params.prefill_assistant,
                          this->params.reasoning_format,
                          this->params.default_template_kwargs,
                          this->chat_formatter_->get_templates(),
                          false,
                          false,
                          false};

  // create the sampler
  LLAMA_LOG_INFO("initializing sampler");
  this->sampler = common_sampler_init(this->model, this->params.sampling);
  if (!this->sampler) {
    LLAMA_LOG_ERROR("Failed to initialize sampler");
    return;
  }

  // check ctx size
  if (this->get_n_ctx() > this->get_n_ctx_train()) {
    LLAMA_LOG_WARN("Model was trained on only %d context tokens (%d "
                   "specified)",
                   this->get_n_ctx_train(), this->get_n_ctx());
  }

  // set initial values
  LLAMA_LOG_INFO("setting initial values");
  if (initial_reset) {
    this->reset();
  }

  // show info
  LLAMA_LOG_INFO("llama.cpp: build = %d, commit = %s", llama_build_number(),
                 llama_commit());
  LLAMA_LOG_INFO("%s", common_params_get_system_info(this->params).c_str());

  LLAMA_LOG_INFO(
      "Generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d",
      this->get_n_ctx(), this->params.n_batch, this->params.n_predict,
      this->params.n_keep);

  if (this->params.grp_attn_n != 1) {
    if (this->params.grp_attn_n > 0) {
      GGML_ASSERT("grp_attn_n must be positive\n");
    }

    if (this->params.grp_attn_w % this->params.grp_attn_n != 0) {
      GGML_ASSERT("grp_attn_w must be a multiple of grp_attn_n\n");
    }
  }

  LLAMA_LOG_INFO(
      "self-extend: n_ctx_train = %d, grp_attn_n = %d, grp_attn_w = %d",
      this->get_n_ctx_train(), this->params.grp_attn_n,
      this->params.grp_attn_w);

  this->init_decision();

  llama_set_embeddings(
      this->ctx,
      this->is_embedding() || this->is_reranking() ||
          (this->is_decision() && this->decision_model_->needs_embeddings()));

  // Initialize speculative decoding if configured
  this->init_speculative();
}

Llama::~Llama() {
  this->canceled = true;

  // Free speculative decoding resources
  if (this->speculative_ != nullptr) {
    common_speculative_print_stats(this->speculative_);
    common_speculative_free(this->speculative_);
    this->speculative_ = nullptr;
  }

  if (this->ctx_dft_ != nullptr) {
    llama_free(this->ctx_dft_);
    this->ctx_dft_ = nullptr;
  }

  if (this->model_dft_ != nullptr) {
    llama_model_free(this->model_dft_);
    this->model_dft_ = nullptr;
  }

  for (ServerSlot &slot : this->server_slots) {
    if (slot.sampler != nullptr) {
      common_sampler_free(slot.sampler);
      slot.sampler = nullptr;
    }
  }

  if (this->sampler != nullptr) {
    common_sampler_free(this->sampler);
    this->sampler = nullptr;
  }

  if (this->gguf_metadata_ != nullptr) {
    gguf_free(this->gguf_metadata_);
    this->gguf_metadata_ = nullptr;
  }

  llama_backend_free();
}

/*
*****************************
*           RESET           *
*           CANCEL          *
*****************************
*/
void Llama::reset() {
  for (ServerSlot &slot : this->server_slots) {
    slot.reset();
    slot.kv_cached_tokens.clear();
    slot.n_kv_cache = 0;
    slot.kv_positions_valid = true;
    slot.checkpoints.clear();
  }

  llama_memory_clear(this->get_memory(), true);
  if (this->ctx_dft_) {
    llama_memory_clear(llama_get_memory(this->ctx_dft_), true);
  }
  if (this->prompt_cache_) {
    this->prompt_cache_->clear();
  }

  this->canceled = false;
  this->n_past = 0;
  this->n_consumed = 0;
  this->ga_i = 0;
}

/*
*****************************
*          METADATA         *
*****************************
*/
std::string Llama::get_metadata(const std::string &key, size_t size) {

  std::vector<char> buffer(size, 0);
  std::string metada_str;

  int32_t res = llama_model_meta_val_str(this->model, key.c_str(),
                                         buffer.data(), buffer.size());
  if (res >= 0 && !buffer.empty()) {
    // llama_model_meta_val_str returns the value length and leaves the rest of
    // the buffer zero-padded, so only keep the actual value. The result can
    // exceed the buffer size when the value is truncated.
    size_t value_size = std::min(static_cast<size_t>(res), buffer.size() - 1);
    metada_str = std::string(buffer.data(), value_size);
  }

  return metada_str;
}

std::string Llama::get_metadata(const std::string &model_name,
                                const std::string &key, size_t size) {
  std::ostringstream model_key;
  model_key << model_name.c_str() << key.c_str();
  std::string value = this->get_metadata(model_key.str(), size);
  return value;
}

std::string Llama::get_metadata_full(const std::string &key) {
  const int32_t length =
      llama_model_meta_val_str(this->model, key.c_str(), nullptr, 0);

  if (length >= 0) {
    std::vector<char> buffer(static_cast<size_t>(length) + 1, 0);
    llama_model_meta_val_str(this->model, key.c_str(), buffer.data(),
                             buffer.size());
    return std::string(buffer.data(), static_cast<size_t>(length));
  }

  // llama_model_meta_* skips GGUF arrays, so fall back to the raw metadata.
  if (this->gguf_metadata_ == nullptr) {
    return "";
  }

  const int64_t key_id = gguf_find_key(this->gguf_metadata_, key.c_str());
  if (key_id < 0) {
    return "";
  }

  return gguf_value_to_string(this->gguf_metadata_, key_id);
}

int Llama::get_int_metadata(const std::string &key, size_t size) {
  return llama_ros::parse_metadata_int(this->get_metadata(key, size));
}

int Llama::get_int_metadata(const std::string &model_name,
                            const std::string &key, size_t size) {
  return llama_ros::parse_metadata_int(
      this->get_metadata(model_name, key, size));
}

float Llama::get_float_metadata(const std::string &key, size_t size) {
  return llama_ros::parse_metadata_float(this->get_metadata(key, size));
}

float Llama::get_float_metadata(const std::string &model_name,
                                const std::string &key, size_t size) {
  return llama_ros::parse_metadata_float(
      this->get_metadata(model_name, key, size));
}

Metadata Llama::get_metadata() {

  std::map<std::string, std::string> gguf_types = {
      {"", ""},
      {"0", "ALL_F32"},
      {"1", "MOSTLY_F16"},
      {"2", "MOSTLY_Q4_0"},
      {"3", "MOSTLY_Q4_1"},
      {"4", "MOSTLY_Q4_1_SOME_F16"},
      {"7", "MOSTLY_Q8_0"},
      {"8", "MOSTLY_Q5_0"},
      {"9", "MOSTLY_Q5_1"},
      {"10", "MOSTLY_Q2_K"},
      {"11", "MOSTLY_Q3_K_S"},
      {"12", "MOSTLY_Q3_K_M"},
      {"13", "MOSTLY_Q3_K_L"},
      {"14", "MOSTLY_Q4_K_S"},
      {"15", "MOSTLY_Q4_K_M"},
      {"16", "MOSTLY_Q5_K_S"},
      {"17", "MOSTLY_Q5_K_M"},
      {"18", "MOSTLY_Q6_K"},
  };

  Metadata metadata;

  // required general metadata
  metadata.general.architecture =
      this->get_metadata_full("general.architecture");
  metadata.general.quantization_version =
      this->get_int_metadata("general.quantization_version", 4);
  metadata.general.alignment = this->get_int_metadata("general.alignment", 4);

  // general metadata
  metadata.general.name = this->get_metadata_full("general.name");
  metadata.general.author = this->get_metadata_full("general.author");
  metadata.general.version = this->get_metadata_full("general.version");
  metadata.general.organization =
      this->get_metadata_full("general.organization");

  metadata.general.basename = this->get_metadata_full("general.basename");
  metadata.general.finetune = this->get_metadata_full("general.finetune");
  metadata.general.description = this->get_metadata_full("general.description");
  metadata.general.quantized_by =
      this->get_metadata_full("general.quantized_by");
  metadata.general.size_label = this->get_metadata_full("general.size_label");

  metadata.general.license = this->get_metadata_full("general.license");
  metadata.general.license_name =
      this->get_metadata_full("general.license.name");
  metadata.general.license_link =
      this->get_metadata_full("general.license.link");

  metadata.general.url = this->get_metadata_full("general.url");
  metadata.general.repo_url = this->get_metadata_full("general.repo_url");
  metadata.general.doi = this->get_metadata_full("general.doi");
  metadata.general.uuid = this->get_metadata_full("general.uuid");

  std::string file_type = this->get_metadata("general.file_type", 32);
  if (gguf_types.find(file_type) != gguf_types.end()) {
    metadata.general.file_type = gguf_types.at(file_type);
  }

  metadata.general.tags =
      llama_ros::parse_metadata_array(this->get_metadata_full("general.tags"));
  metadata.general.languages = llama_ros::parse_metadata_array(
      this->get_metadata_full("general.languages"));
  metadata.general.datasets = llama_ros::parse_metadata_array(
      this->get_metadata_full("general.datasets"));

  // cap the count so a corrupt GGUF cannot trigger an unbounded loop
  const int base_model_count =
      std::min(this->get_int_metadata("general.base_model.count", 16), 64);

  for (int i = 0; i < base_model_count; ++i) {
    Metadata::GeneralInfo::BaseModelInfo base;
    const std::string prefix = "general.base_model." + std::to_string(i) + ".";
    base.name = this->get_metadata_full(prefix + "name");
    if (base.name.empty()) {
      break;
    }
    base.author = this->get_metadata_full(prefix + "author");
    base.version = this->get_metadata_full(prefix + "version");
    base.organization = this->get_metadata_full(prefix + "organization");
    base.repo_url = this->get_metadata_full(prefix + "repo_url");
    metadata.general.base_models.push_back(std::move(base));
  }

  // llm metadata
  metadata.model.context_length = this->get_int_metadata(
      metadata.general.architecture, ".context_length", 16);
  metadata.model.embedding_length = this->get_int_metadata(
      metadata.general.architecture, ".embedding_length", 16);
  metadata.model.block_count =
      this->get_int_metadata(metadata.general.architecture, ".block_count", 16);
  metadata.model.feed_forward_length = this->get_int_metadata(
      metadata.general.architecture, ".feed_forward_length", 16);

  metadata.model.use_parallel_residual =
      this->get_metadata(metadata.general.architecture,
                         ".use_parallel_residual", 16) == "true";
  metadata.model.tensor_data_layout = this->get_metadata_full(
      metadata.general.architecture + ".tensor_data_layout");

  metadata.model.expert_count = this->get_int_metadata(
      metadata.general.architecture, ".expert_count", 16);
  metadata.model.expert_used_count = this->get_int_metadata(
      metadata.general.architecture, ".expert_used_count", 16);

  // llm attention metadata
  metadata.model.attention.head_count = this->get_int_metadata(
      metadata.general.architecture, ".attention.head_count", 16);
  metadata.model.attention.head_count_kv = this->get_int_metadata(
      metadata.general.architecture, ".attention.head_count_kv", 16);

  metadata.model.attention.max_alibi_bias = this->get_float_metadata(
      metadata.general.architecture, ".attention.max_alibi_bias", 32);
  metadata.model.attention.clamp_kqv = this->get_float_metadata(
      metadata.general.architecture, ".attention.clamp_kqv", 32);

  metadata.model.attention.layer_norm_epsilon = this->get_float_metadata(
      metadata.general.architecture, ".attention.layer_norm_epsilon", 32);
  metadata.model.attention.layer_norm_rms_epsilon = this->get_float_metadata(
      metadata.general.architecture, ".attention.layer_norm_rms_epsilon", 16);

  metadata.model.attention.key_length = this->get_int_metadata(
      metadata.general.architecture, ".attention.key_length", 16);
  metadata.model.attention.value_length = this->get_int_metadata(
      metadata.general.architecture, ".attention.value_length", 16);

  // rope metadata
  metadata.model.rope.dimension_count = this->get_int_metadata(
      metadata.general.architecture, ".rope.dimension_count", 16);
  metadata.model.rope.freq_base = this->get_float_metadata(
      metadata.general.architecture, ".rope.freq_base", 16);

  metadata.model.rope.scaling_type = this->get_metadata_full(
      metadata.general.architecture + ".rope.scaling.type");
  metadata.model.rope.scaling_factor = this->get_float_metadata(
      metadata.general.architecture, ".rope.scaling.factor", 16);
  metadata.model.rope.scaling_original_context_length =
      this->get_int_metadata(metadata.general.architecture,
                             ".rope.scaling.original_context_length", 16);
  metadata.model.rope.scaling_finetuned =
      this->get_metadata(metadata.general.architecture,
                         ".rope.scaling.finetuned", 8) == "true";

  // tokenizer metadata
  metadata.tokenizer.model = this->get_metadata_full("tokenizer.ggml.model");

  metadata.tokenizer.bos_token_id =
      this->get_int_metadata("tokenizer.ggml.bos_token_id", 16);
  metadata.tokenizer.eos_token_id =
      this->get_int_metadata("tokenizer.ggml.eos_token_id", 16);
  metadata.tokenizer.unknown_token_id =
      this->get_int_metadata("tokenizer.ggml.unknown_token_id", 16);
  metadata.tokenizer.padding_token_id =
      this->get_int_metadata("tokenizer.ggml.padding_token_id", 16);
  metadata.tokenizer.separator_token_id =
      this->get_int_metadata("tokenizer.ggml.separator_token_id", 16);

  metadata.tokenizer.add_bos_token =
      this->get_metadata("tokenizer.ggml.add_bos_token", 8) == "true";
  metadata.tokenizer.add_eos_token =
      this->get_metadata("tokenizer.ggml.add_eos_token", 8) == "true";
  const int mask_token_id =
      this->get_int_metadata("tokenizer.ggml.mask_token_id", 16);
  metadata.tokenizer.mask_token_id =
      mask_token_id >= 0 ? static_cast<uint32_t>(mask_token_id) : 0;

  // named templates are stored as tokenizer.chat_template.<name>
  const std::string prefix_template = "tokenizer.chat_template.";
  for (int32_t i = 0; i < llama_model_meta_count(this->model); i++) {
    char key[256];
    if (llama_model_meta_key_by_index(this->model, i, key, sizeof(key)) < 0 ||
        !string_starts_with(key, prefix_template)) {
      continue;
    }
    metadata.tokenizer.chat_templates.push_back(key + prefix_template.size());
  }

  metadata.tokenizer.chat_template =
      this->get_metadata_full("tokenizer.chat_template");

  metadata.sampling.sequence = llama_ros::split_semicolon(
      this->get_metadata_full("general.sampling.sequence"));
  metadata.sampling.top_k =
      this->get_int_metadata("general.sampling.top_k", 16);
  metadata.sampling.top_p =
      this->get_float_metadata("general.sampling.top_p", 16);
  metadata.sampling.min_p =
      this->get_float_metadata("general.sampling.min_p", 16);
  metadata.sampling.xtc_probability =
      this->get_float_metadata("general.sampling.xtc_probability", 16);
  metadata.sampling.xtc_threshold =
      this->get_float_metadata("general.sampling.xtc_threshold", 16);
  metadata.sampling.temp =
      this->get_float_metadata("general.sampling.temp", 16);
  metadata.sampling.penalty_last_n =
      this->get_int_metadata("general.sampling.penalty_last_n", 16);
  metadata.sampling.penalty_repeat =
      this->get_float_metadata("general.sampling.penalty_repeat", 16);
  metadata.sampling.mirostat =
      this->get_int_metadata("general.sampling.mirostat", 16);
  metadata.sampling.mirostat_tau =
      this->get_float_metadata("general.sampling.mirostat_tau", 16);
  metadata.sampling.mirostat_eta =
      this->get_float_metadata("general.sampling.mirostat_eta", 16);

  const std::string arch = metadata.general.architecture;
  metadata.decision.enabled = this->is_decision();
  metadata.decision.type =
      metadata.decision.enabled
          ? this->get_metadata_full(arch + ".decision.type")
          : "";
  metadata.decision.max_head_tokens =
      metadata.decision.enabled ? static_cast<uint32_t>(this->get_int_metadata(
                                      arch + ".decision.max_head_tokens", 16))
                                : 0u;

  if (metadata.decision.enabled) {
    const std::string prefix_temp = arch + ".decision.temperature.";

    for (int32_t i = 0; i < llama_model_meta_count(this->model); i++) {
      char key[256];
      if (llama_model_meta_key_by_index(this->model, i, key, sizeof(key)) < 0 ||
          !string_starts_with(key, prefix_temp)) {
        continue;
      }
      metadata.decision.temperature_names.push_back(key + prefix_temp.size());
      metadata.decision.temperatures.push_back(
          llama_ros::parse_metadata_float(this->get_metadata_full(key), 1.0f));
    }
  }

  const char *systemone = llama_model_chat_template(this->model, "systemone");
  metadata.decision.systemone_template =
      metadata.decision.enabled && systemone != nullptr ? std::string(systemone)
                                                        : "";

  return metadata;
}

/*
*****************************
*          TOKENIZE         *
*         DETOKENIZE        *
*****************************
*/
std::vector<llama_token> Llama::tokenize(const std::string &text, bool add_bos,
                                         bool special) {
  return common_tokenize(this->get_vocab(), text, add_bos, special);
}

std::string Llama::detokenize(const std::vector<llama_token> &tokens) {
  std::string text;

  for (llama_token t : tokens) {
    if (t == LLAMA_TOKEN_NULL)
      continue;
    text.append(common_token_to_piece(this->ctx, t));
  }

  return text;
}

void Llama::cancel() {
  this->canceled = true;
  this->task_registry_->fail_all_pending();
}

bool Llama::supports_precompute() const {
  return this->params.n_parallel == 1 && this->params.cache_prompt &&
         !this->params.embedding && this->params.mmproj.path.empty() &&
         !this->is_speculative() && !llama_model_is_recurrent(this->model) &&
         !llama_model_is_hybrid(this->model) && this->partial_seq_removal_ &&
         llama_model_n_swa(this->model) == 0;
}

void Llama::cancel_goal(uint64_t goal_id) {
  this->task_registry_->request_cancel(goal_id);
}

/*
*******************************
*         EMBEDDINGS          *
*******************************
*/
Result<llama_ros::ServerTaskResultEmbedding>
Llama::generate_embeddings(const std::string &text) {
  // Validate input text is not empty
  if (text.empty()) {
    return Result<ServerTaskResultEmbedding>::error(
        "Input text cannot be empty for embedding generation");
  }

  auto slot = this->slot_manager_->wait_for_available_slot();
  if (!slot) {
    return Result<ServerTaskResultEmbedding>::error(
        "No slot available for embedding generation");
  }

  const uint64_t gid = llama_utils::generate_random_uint64();
  slot->goal_id = gid;
  auto fut = this->task_registry_->register_pending(gid);

  this->embedding_handler_->handle(text, slot);

  try {
    auto result = fut.get();

    if (auto *out = dynamic_cast<ServerTaskResultEmbedding *>(result.get())) {
      return Result<ServerTaskResultEmbedding>::ok(*out);
    }
    return Result<ServerTaskResultEmbedding>::error(
        "Invalid result type returned");
  } catch (const std::exception &e) {
    return Result<ServerTaskResultEmbedding>::error(
        std::string("Exception during embedding generation: ") + e.what());
  }
}

/*
*****************************
*         RERANKING         *
*****************************
*/
Result<std::vector<llama_ros::ServerTaskResultRerank>>
Llama::rank_documents(const std::string &query,
                      const std::vector<std::string> &documents) {
  if (!this->is_reranking()) {
    return Result<std::vector<ServerTaskResultRerank>>::error(
        "Llama must be created with reranking enabled to perform reranking");
  }

  // Register all tasks
  auto n_documents = documents.size();
  std::unordered_map<uint64_t, std::future<ServerTaskResultPtr>> futs(
      n_documents);

  const uint64_t slot_gid = llama_utils::generate_random_uint64();

  for (size_t i = 0; i < documents.size(); ++i) {
    auto slot = this->slot_manager_->wait_for_available_slot();

    const uint64_t gid =
        (static_cast<uint64_t>(slot_gid) << 32) | static_cast<uint64_t>(i);

    slot->goal_id = gid;
    auto fut = this->task_registry_->register_pending(gid);

    LLAMA_LOG_INFO(
        "Submitting rerank task %lu for document %zu (slot goal_id: %lu)", gid,
        i, slot_gid);

    this->rerank_handler_->handle(query, documents[i], slot);

    futs.emplace(gid, std::move(fut));
  }

  std::vector<llama_ros::ServerTaskResultRerank> results;
  results.reserve(documents.size());

  size_t n_collected = 0;
  while (n_collected < n_documents) {
    uint64_t first_gid = this->task_registry_->wait_for_done();

    if (auto it = futs.find(first_gid); it != futs.end()) {
      ServerTaskResultPtr ptr = it->second.get();

      if (auto *out = dynamic_cast<ServerTaskResultRerank *>(ptr.get())) {
        results.push_back(*out);
      } else {
        LLAMA_LOG_ERROR(
            "Failed to cast ServerTaskResultPtr to ServerTaskResultRerank");
      }

      futs.erase(it);
      n_collected++;
    }

    while (this->task_registry_->has_done_tasks()) {
      uint64_t gid = this->task_registry_->wait_for_done();

      if (auto it = futs.find(gid); it != futs.end()) {
        ServerTaskResultPtr ptr = it->second.get();

        if (auto *out = dynamic_cast<ServerTaskResultRerank *>(ptr.get())) {
          results.push_back(*out);
        } else {
          LLAMA_LOG_ERROR(
              "Failed to cast ServerTaskResultPtr to ServerTaskResultRerank");
        }

        futs.erase(it);
        n_collected++;
      }
    }
  }

  std::sort(
      results.begin(), results.end(),
      [](const llama_ros::ServerTaskResultRerank &a,
         const llama_ros::ServerTaskResultRerank &b) { return a.id < b.id; });

  return Result<std::vector<ServerTaskResultRerank>>::ok(std::move(results));
}

/*
*******************************
*          DECISIONS          *
*******************************
*/
void Llama::init_decision() {
  this->decision_model_ = std::make_unique<DecisionModel>();
  this->decision_model_->init_from_model(this->model);
  if (!this->decision_model_->enabled()) {
    this->decision_model_.reset();
    return;
  }
  LLAMA_LOG_INFO("Decision model enabled (type %d)",
                 (int)this->decision_model_->type());
}

common_json Llama::parse_decision_state(const std::string &state) {
  try {
    return common_json::parse(state);
  } catch (const std::exception &) {
    return common_json(state);
  }
}

std::string Llama::validate_decision_question(const DecisionQuestion &question,
                                              size_t n_options_max) {
  switch (question.type) {
  case DECISION_QUESTION_CHOICE: {
    if (question.keys.empty()) {
      return "choice questions need at least one option key";
    }
    if (!question.descriptions.empty() &&
        question.descriptions.size() != question.keys.size()) {
      return "keys and descriptions must have the same size";
    }
    for (const auto &key : question.keys) {
      if (key.empty()) {
        return "option keys must not be empty";
      }
    }
    if (question.keys.size() > n_options_max) {
      return string_format(
          "too many options (%zu), this model supports at most %zu",
          question.keys.size(), n_options_max);
    }
    return "";
  }
  case DECISION_QUESTION_SCORE: {
    if (question.descriptions.size() < 2 || question.descriptions.size() > 10) {
      return "score questions need between 2 and 10 levels";
    }
    if (question.descriptions.size() > n_options_max) {
      return string_format(
          "too many options (%zu), this model supports at most %zu",
          question.descriptions.size(), n_options_max);
    }
    return "";
  }
  case DECISION_QUESTION_NOUL: {
    if (!question.descriptions.empty() && question.descriptions.size() != 2) {
      return "noul questions take at most two descriptions (false, true)";
    }
    if (n_options_max < 2) {
      return string_format(
          "too many options (2), this model supports at most %zu",
          n_options_max);
    }
    return "";
  }
  }
  return "unknown decision question type";
}

std::vector<Result<DecisionAnswer>>
Llama::evaluate_decisions(const std::string &state,
                          const std::vector<DecisionQuestion> &questions,
                          size_t n_images) {
  std::vector<Result<DecisionAnswer>> results;
  results.reserve(questions.size());

  const std::string no_model =
      "Llama must be created with a decision model to evaluate questions";
  if (!this->is_decision()) {
    for (size_t i = 0; i < questions.size(); ++i) {
      results.push_back(Result<DecisionAnswer>::error(no_model));
    }
    return results;
  }

  std::vector<DecisionQuestion> qs = questions;
  for (size_t i = 0; i < qs.size(); ++i) {
    if (qs[i].id.empty()) {
      qs[i].id = std::to_string(i);
    }
  }

  const common_json state_json = parse_decision_state(state);

  if (this->decision_model_->is_joint()) {
    return this->evaluate_decisions_joint(state_json, qs, n_images);
  }

  for (const auto &question : qs) {
    try {
      results.push_back(
          this->evaluate_decision(state_json, qs, question, n_images));
    } catch (const std::exception &e) {
      results.push_back(Result<DecisionAnswer>::error(e.what()));
    }
  }
  return results;
}

Result<DecisionAnswer>
Llama::evaluate_decision(const common_json &state,
                         const std::vector<DecisionQuestion> &questions,
                         const DecisionQuestion &question, size_t n_images) {
  const std::string validation = validate_decision_question(
      question, this->decision_model_->n_options_max());
  if (!validation.empty()) {
    return Result<DecisionAnswer>::error(validation);
  }

  const auto options = this->decision_model_->options_from_question(question);
  const size_t n_variants =
      this->decision_model_->n_variants(question, options);

  std::vector<std::vector<float>> scores;
  scores.reserve(n_variants);

  for (size_t variant = 0; variant < n_variants; ++variant) {
    std::string prompt;
    try {
      prompt = this->decision_model_->render(state, questions, question,
                                             options, variant, n_images);
    } catch (const std::exception &e) {
      return Result<DecisionAnswer>::error(
          std::string("Failed to render the decision prompt: ") + e.what());
    }

    auto slot = this->slot_manager_->wait_for_available_slot();
    if (!slot) {
      return Result<DecisionAnswer>::error(
          "No slot available for decision evaluation");
    }

    const uint64_t gid = llama_utils::generate_random_uint64();
    slot->goal_id = gid;
    auto fut = this->task_registry_->register_pending(gid);

    try {
      this->prepare_decision_slot(prompt, question, options, n_images, slot);
    } catch (const std::exception &e) {
      this->fail_pending(gid, e.what());
      this->release_slot(slot);
      return Result<DecisionAnswer>::error(e.what());
    }

    try {
      auto result = fut.get();
      if (auto *out = dynamic_cast<ServerTaskResultDecision *>(result.get())) {
        scores.push_back(out->scores);
      } else {
        return Result<DecisionAnswer>::error("Invalid result type returned");
      }
    } catch (const std::exception &e) {
      return Result<DecisionAnswer>::error(
          std::string("Exception during decision evaluation: ") + e.what());
    }
  }

  try {
    return Result<DecisionAnswer>::ok(
        this->decision_model_->format_answer(question, options, scores));
  } catch (const std::exception &e) {
    return Result<DecisionAnswer>::error(e.what());
  }
}

std::vector<Result<DecisionAnswer>>
Llama::evaluate_decisions_joint(const common_json &state,
                                const std::vector<DecisionQuestion> &questions,
                                size_t n_images) {
  std::vector<Result<DecisionAnswer>> results;
  results.reserve(questions.size());

  const auto fail_all = [&results, &questions](const std::string &error) {
    results.clear();
    for (size_t i = 0; i < questions.size(); ++i) {
      results.push_back(Result<DecisionAnswer>::error(error));
    }
  };

  if (n_images > 0 && !this->decision_model_->supports_images()) {
    fail_all("images are not supported by this decision model");
    return results;
  }

  for (const auto &question : questions) {
    const std::string validation = validate_decision_question(
        question, this->decision_model_->n_options_max());
    if (!validation.empty()) {
      fail_all(validation);
      return results;
    }
  }

  std::string prompt;
  try {
    prompt = this->decision_model_->render_joint(state, questions, n_images);
  } catch (const std::exception &e) {
    fail_all(std::string("Failed to render the decision prompt: ") + e.what());
    return results;
  }

  auto slot = this->slot_manager_->wait_for_available_slot();
  if (!slot) {
    fail_all("No slot available for decision evaluation");
    return results;
  }

  const uint64_t gid = llama_utils::generate_random_uint64();
  slot->goal_id = gid;
  auto fut = this->task_registry_->register_pending(gid);

  try {
    this->prepare_joint_decision_slot(prompt, questions, n_images, slot);
  } catch (const std::exception &e) {
    this->fail_pending(gid, e.what());
    this->release_slot(slot);
    fail_all(e.what());
    return results;
  }

  std::vector<float> scores;
  try {
    auto result = fut.get();
    if (auto *out = dynamic_cast<ServerTaskResultDecision *>(result.get())) {
      scores = out->scores;
    } else {
      fail_all("Invalid result type returned");
      return results;
    }
  } catch (const std::exception &e) {
    fail_all(std::string("Exception during decision evaluation: ") + e.what());
    return results;
  }

  size_t offset = 0;
  for (const auto &question : questions) {
    const auto options = this->decision_model_->options_from_question(question);
    const size_t n = this->decision_model_->n_outputs(question, options);
    if (offset + n > scores.size()) {
      results.push_back(Result<DecisionAnswer>::error(
          "decision result does not match the number of options"));
      offset += n;
      continue;
    }
    const std::vector<float> question_scores(scores.begin() + offset,
                                             scores.begin() + offset + n);
    offset += n;
    try {
      results.push_back(
          Result<DecisionAnswer>::ok(this->decision_model_->format_answer(
              question, options, {question_scores})));
    } catch (const std::exception &e) {
      results.push_back(Result<DecisionAnswer>::error(e.what()));
    }
  }

  return results;
}

void Llama::prepare_decision_slot(const std::string &prompt,
                                  const DecisionQuestion &question,
                                  const std::vector<DecisionOption> &options,
                                  size_t n_images, ServerSlot *slot) {
  if (n_images > 0) {
    throw std::runtime_error("images are not supported by this node");
  }

  std::vector<llama_token> tokens =
      common_tokenize(this->get_vocab(), prompt, false, true);
  DecisionTaskMeta meta;
  this->decision_model_->fill_task(tokens, question, options, meta);

  if (tokens.size() > (size_t)llama_n_batch(this->ctx)) {
    throw std::runtime_error(
        "The question, its options and the state must fit in one batch; "
        "increase context.n_batch");
  }

  this->decision_handler_->handle(tokens, meta, slot);
}

void Llama::prepare_joint_decision_slot(
    const std::string &prompt, const std::vector<DecisionQuestion> &questions,
    size_t n_images, ServerSlot *slot) {

  if (n_images > 0) {
    throw std::runtime_error("images are not supported by this node");
  }

  std::vector<llama_token> tokens;
  DecisionTaskMeta meta;
  this->decision_model_->fill_task_joint(this->get_vocab(), questions, prompt,
                                         0, tokens, meta);

  if (tokens.size() >
      (size_t)std::min(llama_n_batch(this->ctx), llama_n_ubatch(this->ctx))) {
    throw std::runtime_error(
        "The question, its options and the state must fit in one batch; "
        "increase context.n_batch and context.n_ubatch");
  }

  this->decision_handler_->handle(tokens, meta, slot);
}

bool Llama::process_decision_mtmd_batch(ServerSlot *slot) {
  (void)slot;
  return false;
}

void Llama::handle_prefilled_decision_req(const DecisionTaskMeta &meta,
                                          ServerSlot *slot) {
  this->decision_handler_->handle_prefilled(meta, slot);
}

void Llama::send_decision_result(ServerSlot *slot, int32_t off,
                                 int32_t n_tokens) {
  auto result = std::make_unique<ServerTaskResultDecision>();
  result->id_slot = slot->id;
  result->id = slot->goal_id;
  result->n_tokens = n_tokens;

  const auto &meta = slot->decision;

  // label types: logits of one token per option, at the last prompt token
  if (!meta.labels.empty()) {
    const float *logits = llama_get_logits_ith(this->ctx, slot->i_batch - off);
    if (logits == nullptr) {
      this->fail_pending(slot->goal_id, "Failed to get decision logits");
      return;
    }
    for (const llama_token label : meta.labels) {
      result->scores.push_back(logits[label]);
    }
    const auto id = result->id;
    this->fulfill_pending(id, std::move(result));
    return;
  }

  // joint head (clef): the scores are the first rows of the embeddings
  if (!meta.order.empty()) {
    int32_t count = 0;
    for (int32_t i = 0; i < n_tokens && count < meta.n_scores; ++i) {
      const auto &batch_token = this->batch.tokens[off + i];
      if (!batch_token.output || batch_token.seq_id != slot->id) {
        continue;
      }
      const float *embd = llama_get_embeddings_ith(this->ctx, i);
      if (embd == nullptr) {
        this->fail_pending(slot->goal_id,
                           "Failed to get decision embeddings, the question "
                           "and its options must fit in one batch");
        return;
      }
      result->scores.push_back(embd[0]);
      count++;
    }
    if (count != meta.n_scores) {
      this->fail_pending(slot->goal_id,
                         "Failed to read all decision scores, the question "
                         "and its options must fit in one batch");
      return;
    }
    const auto id = result->id;
    this->fulfill_pending(id, std::move(result));
    return;
  }

  // marker types: laya reads embeddings[column], kev dot-products the pointer
  // with each marker
  const int32_t n_embd_out = llama_model_n_embd_out(this->model);
  const int32_t n_pointer = n_embd_out / 2;
  const float *embd_q = nullptr;

  if (meta.pointer >= 0) {
    for (int32_t i = n_tokens - 1; i >= 0; --i) {
      const auto &batch_token = this->batch.tokens[off + i];
      if (batch_token.output && batch_token.seq_id == slot->id) {
        embd_q = llama_get_embeddings_ith(this->ctx, i);
        break;
      }
    }
    if (embd_q == nullptr) {
      this->fail_pending(slot->goal_id,
                         "Failed to get decision embeddings, the question "
                         "and its options must fit in one batch");
      return;
    }
  }

  size_t n_expected = 0;
  for (const auto token : slot->prompt_tokens) {
    if (token == this->decision_model_->get_token_marker()) {
      n_expected++;
    }
  }

  for (int32_t i = 0; i < n_tokens; ++i) {
    const auto &batch_token = this->batch.tokens[off + i];
    if (!batch_token.output || batch_token.seq_id != slot->id ||
        batch_token.id != this->decision_model_->get_token_marker()) {
      continue;
    }

    const float *embd = llama_get_embeddings_ith(this->ctx, i);
    if (embd == nullptr) {
      this->fail_pending(slot->goal_id,
                         "Failed to get decision embeddings, the question "
                         "and its options must fit in one batch");
      return;
    }

    if (meta.pointer < 0) {
      if (meta.column < 0 || meta.column >= n_embd_out) {
        this->fail_pending(slot->goal_id, "Invalid decision question type");
        return;
      }
      result->scores.push_back(embd[meta.column]);
      continue;
    }

    float dot = 0.0f;
    for (int32_t j = 0; j < n_pointer; j++) {
      dot += embd_q[j] * embd[n_pointer + j];
    }
    result->scores.push_back(dot / sqrtf((float)n_pointer));
  }

  if (result->scores.size() != n_expected) {
    this->fail_pending(slot->goal_id,
                       "Failed to read all decision option scores, the "
                       "question and its options must fit in one batch");
    return;
  }

  const auto id = result->id;
  this->fulfill_pending(id, std::move(result));
}

/*
*******************************
*            LORAS            *
*******************************
*/
std::vector<LoRA> Llama::list_loras() {

  // LoRA adapters are read-only here, no lock needed
  std::vector<LoRA> loras;

  for (size_t i = 0; i < this->lora_adapters.size(); ++i) {
    auto &lora_i = this->lora_adapters[i];

    LoRA lora_aux;
    lora_aux.id = i;
    lora_aux.path = lora_i.path;
    lora_aux.scale = lora_i.scale;

    loras.push_back(lora_aux);
  }

  return loras;
}

void Llama::update_loras(std::vector<LoRA> loras) {

  // LoRA updates are thread-safe at llama.cpp level
  for (auto lora : loras) {
    if (lora.id >= 0 && lora.id < (int)this->lora_adapters.size()) {

      LLAMA_LOG_INFO("Updating LoRA (%d: '%s') from %f to %f", lora.id,
                     this->lora_adapters[lora.id].path.c_str(),
                     this->lora_adapters[lora.id].scale, lora.scale);

      float scale = lora.scale;

      if (scale < 0.0) {
        LLAMA_LOG_WARN("Scale %f cannot be lower than 0.0, setting it to 0.0",
                       scale);
        scale = 0.0;

      } else if (scale > 1.0) {
        LLAMA_LOG_WARN("Scale %f cannot be greater than 1.0, setting it to 1.0",
                       scale);
        scale = 1.0;
      }

      this->lora_adapters[lora.id].scale = scale;

    } else {
      LLAMA_LOG_ERROR("Invalid LoRA id: %d", lora.id);
    }
  }

  common_set_adapter_lora(this->ctx, this->lora_adapters);
}

/*
*****************************
*     GENERATE RESPONSE     *
*****************************
*/
Result<ServerTaskResultCompletion>
Llama::generate_response(int slot_gid, const std::string &input_prompt,
                         common_params_sampling sparams,
                         ServerSlot::GenerateResponseCallback callback,
                         std::vector<std::string> stop, bool reset,
                         bool precompute) {
  auto slot = this->slot_manager_->get_slot_by_gid(slot_gid);
  if (!slot) {
    return Result<ServerTaskResultCompletion>::error(
        "Slot not found for given ID");
  }

  if (precompute && (reset || !this->supports_precompute())) {
    this->release_slot(slot);
    return Result<ServerTaskResultCompletion>::error(
        "Precompute requires one text slot, prompt caching, reset=false and a "
        "non-recurrent, non-hybrid, non-speculative model");
  }
  slot->precompute = precompute;
  auto fut = this->task_registry_->register_pending(slot_gid);

  this->handle_completion_req(input_prompt, slot, sparams, callback, stop,
                              reset);

  try {
    auto result = fut.get();

    if (auto *out = dynamic_cast<ServerTaskResultCompletion *>(result.get())) {
      return Result<ServerTaskResultCompletion>::ok(*out);
    }

    return Result<ServerTaskResultCompletion>::error(
        "Invalid result type returned");

  } catch (const std::exception &e) {
    return Result<ServerTaskResultCompletion>::error(
        std::string("Exception during response generation: ") + e.what());
  }
}

Result<ServerTaskResultCompletion>
Llama::generate_chat_response(int slot_gid,
                              llama_utils::ChatCompletionsContext chat_context,
                              ServerSlot::GenerateResponseCallback callback) {
  auto slot = this->slot_manager_->get_slot_by_gid(slot_gid);
  if (!slot) {
    return Result<ServerTaskResultCompletion>::error(
        "Slot not found for given ID");
  }

  auto fut = this->task_registry_->register_pending(slot_gid);

  this->handle_chat_completion_req(chat_context, slot, callback);

  try {
    auto result = fut.get();

    if (auto *out = dynamic_cast<ServerTaskResultCompletion *>(result.get())) {
      return Result<ServerTaskResultCompletion>::ok(*out);
    }

    return Result<ServerTaskResultCompletion>::error(
        "Invalid result type returned");

  } catch (const std::exception &e) {
    return Result<ServerTaskResultCompletion>::error(
        std::string("Exception during chat response generation: ") + e.what());
  }
}

/*
*****************************
*          SAMPLE           *
*****************************
*/
std::vector<TokenProb> Llama::get_probs(ServerSlot *slot) {
  std::vector<TokenProb> probs;

  const auto *cur_p = common_sampler_get_candidates(slot->sampler, true);

  const int32_t n_probs = slot->params.sampling.n_probs;

  for (int i = 0; i < n_probs; ++i) {
    probs.push_back({
        cur_p->data[i].id,
        (size_t)i >= cur_p->size ? 0.0f : cur_p->data[i].p,
    });
  }

  return probs;
}

std::vector<SelectedLogProb>
Llama::convert_probs_to_logprobs(ServerSlot *slot) {
  std::vector<SelectedLogProb> result;

  // Convert each token's probability data
  for (size_t i = 0; i < slot->generated_probs.size(); ++i) {
    const auto &token_probs = slot->generated_probs[i];

    if (token_probs.empty()) {
      continue;
    }

    SelectedLogProb selected;

    // First entry is the chosen token
    selected.chosen_token.token = token_probs[0].token;
    selected.chosen_token.probability = std::log(token_probs[0].probability);
    selected.chosen_token.text =
        common_token_to_piece(this->ctx, token_probs[0].token);

    // Add all alternatives (including the chosen one)
    for (const auto &tp : token_probs) {
      LogProb lp;
      lp.token = tp.token;
      lp.probability = std::log(tp.probability);
      lp.text = common_token_to_piece(this->ctx, tp.token);
      selected.data.push_back(lp);
    }

    result.push_back(selected);
  }

  return result;
}

/*
*****************************
*   CHAT COMPLETION FUNCS   *
*****************************
*/
llama_perf_context_data Llama::get_perf_data() {
  return llama_perf_context(this->ctx);
}

common_chat_params Llama::get_chat_params(common_chat_templates *tmpls,
                                          common_chat_templates_inputs inputs) {
  return common_chat_templates_apply(tmpls, inputs);
}

void Llama::release_slot(ServerSlot *slot) {
  this->task_registry_->clear_cancel(slot->goal_id);
  this->slot_manager_->release_slot(slot);
}

void Llama::maybe_create_checkpoint(ServerSlot &slot) {
  if (!this->checkpoints_enabled_ ||
      slot.task_type != SERVER_TASK_TYPE_COMPLETION || slot.n_past <= 0) {
    return;
  }

  const int64_t last_tokens =
      slot.checkpoints.empty() ? -1 : slot.checkpoints.back().n_tokens;
  if (last_tokens >= 0 &&
      slot.n_past - last_tokens < this->params.checkpoint_min_step) {
    return;
  }

  auto *mem = llama_get_memory(this->ctx);
  const llama_pos pos_max = llama_memory_seq_pos_max(mem, slot.id);
  if (pos_max < 0) {
    return;
  }

  common_prompt_checkpoint ckpt;
  ckpt.update_pos(slot.n_past, llama_memory_seq_pos_min(mem, slot.id), pos_max);
  if (!save_seq_state(ckpt.data_tgt, this->ctx, slot.id,
                      LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
    LLAMA_LOG_WARN("Slot %d: failed to snapshot checkpoint state", slot.id);
    return;
  }

  if (!save_seq_state(ckpt.data_dft, this->ctx_dft_, slot.id,
                      LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
    LLAMA_LOG_WARN("Slot %d: failed to snapshot draft checkpoint state",
                   slot.id);
    return;
  }

  if (this->speculative_ != nullptr) {
    common_speculative_get_state(this->speculative_, slot.id, ckpt.data_spec);
  }

  LLAMA_LOG_DEBUG("Slot %d: created context checkpoint at %d tokens (%.3f MiB)",
                  slot.id, slot.n_past, (float)ckpt.size() / 1024.0f / 1024.0f);

  slot.add_checkpoint(std::move(ckpt), this->params.n_ctx_checkpoints);
}

bool Llama::try_restore_checkpoint(
    ServerSlot &slot, const std::vector<llama_token> &prompt_tokens,
    size_t &reused) {
  if (!this->checkpoints_enabled_ || slot.kv_positions_valid ||
      slot.checkpoints.empty() || slot.kv_cached_tokens.empty()) {
    return false;
  }

  const size_t common = slot.common_prefix_len(prompt_tokens);
  const common_prompt_checkpoint *ckpt =
      slot.find_checkpoint(static_cast<int64_t>(common));
  if (ckpt == nullptr) {
    return false;
  }

  auto *mem = llama_get_memory(this->ctx);
  llama_memory_seq_rm(mem, slot.id, -1, -1);

  if (!load_seq_state(ckpt->data_tgt, this->ctx, slot.id,
                      LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) ||
      !load_seq_state(ckpt->data_dft, this->ctx_dft_, slot.id,
                      LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
    LLAMA_LOG_WARN("Slot %d: failed to restore context checkpoint", slot.id);
    return false;
  }

  if (this->speculative_ != nullptr) {
    common_speculative_set_state(this->speculative_, slot.id, ckpt->data_spec);
  }

  reused = std::min(static_cast<size_t>(ckpt->n_tokens), common);
  if (reused == prompt_tokens.size() && reused > 0) {
    reused--;
  }

  slot.n_past = static_cast<int32_t>(reused);
  slot.n_kv_cache = static_cast<int32_t>(reused);
  slot.kv_positions_valid = true;

  LLAMA_LOG_INFO("Slot %d: restored context checkpoint (%d tokens, %zu "
                 "reusable)",
                 slot.id, (int)ckpt->n_tokens, reused);

  return reused > 0;
}

bool Llama::try_load_prompt_cache(ServerSlot &slot,
                                  const std::vector<llama_token> &prompt_tokens,
                                  size_t &reused) {
  if (this->prompt_cache_ == nullptr || !this->prompt_cache_->enabled() ||
      !slot.map_pos_to_media.empty()) {
    return false;
  }

  const PromptCacheEntry *best = this->prompt_cache_->find_best(prompt_tokens);
  if (best == nullptr) {
    return false;
  }

  auto *mem = llama_get_memory(this->ctx);
  llama_memory_seq_rm(mem, slot.id, -1, -1);

  if (!load_seq_state(best->state.data_tgt, this->ctx, slot.id,
                      LLAMA_STATE_SEQ_FLAGS_NONE) ||
      !load_seq_state(best->state.data_dft, this->ctx_dft_, slot.id,
                      LLAMA_STATE_SEQ_FLAGS_NONE)) {
    LLAMA_LOG_WARN("Slot %d: failed to restore prompt cache state", slot.id);
    this->prompt_cache_->erase(best);
    return false;
  }

  if (this->speculative_ != nullptr) {
    common_speculative_set_state(this->speculative_, slot.id,
                                 best->state.data_spec);
  }

  reused = common_prefix_len(best->tokens, prompt_tokens);
  if (reused == prompt_tokens.size() && reused > 0) {
    reused--;
  }

  slot.kv_cached_tokens = best->tokens;
  slot.n_kv_cache = static_cast<int32_t>(best->tokens.size());
  slot.n_past = static_cast<int32_t>(reused);
  slot.kv_positions_valid = true;

  LLAMA_LOG_INFO("Slot %d: restored %zu/%zu tokens from prompt cache", slot.id,
                 reused, prompt_tokens.size());

  this->prompt_cache_->erase(best);
  return reused > 0;
}

void Llama::save_prompt_to_cache(ServerSlot &slot) {
  if (this->prompt_cache_ == nullptr || !this->prompt_cache_->enabled() ||
      slot.task_type != SERVER_TASK_TYPE_COMPLETION ||
      !slot.map_pos_to_media.empty() || !slot.kv_positions_valid ||
      slot.n_kv_cache <= 0 || slot.kv_cached_tokens.empty()) {
    return;
  }

  std::vector<llama_token> tokens = slot.kv_cached_tokens;
  tokens.insert(tokens.end(), slot.generated_tokens.begin(),
                slot.generated_tokens.end());

  tokens.resize(std::min(tokens.size(), static_cast<size_t>(slot.n_kv_cache)));

  PromptCacheEntry *entry = this->prompt_cache_->alloc(tokens);
  if (entry == nullptr) {
    return;
  }

  auto *mem = llama_get_memory(this->ctx);
  if (llama_memory_seq_pos_max(mem, slot.id) < 0) {
    LLAMA_LOG_DEBUG("Slot %d: nothing to save, KV is empty", slot.id);
    this->prompt_cache_->erase(entry);
    return;
  }

  if (!save_seq_state(entry->state.data_tgt, this->ctx, slot.id,
                      LLAMA_STATE_SEQ_FLAGS_NONE)) {
    LLAMA_LOG_WARN("Slot %d: failed to snapshot state, skipping prompt cache",
                   slot.id);
    this->prompt_cache_->erase(entry);
    return;
  }

  if (!save_seq_state(entry->state.data_dft, this->ctx_dft_, slot.id,
                      LLAMA_STATE_SEQ_FLAGS_NONE)) {
    LLAMA_LOG_WARN("Slot %d: failed to snapshot draft state, skipping prompt "
                   "cache",
                   slot.id);
    this->prompt_cache_->erase(entry);
    return;
  }

  if (this->speculative_ != nullptr) {
    common_speculative_get_state(this->speculative_, slot.id,
                                 entry->state.data_spec);
  }

  this->prompt_cache_->update();

  LLAMA_LOG_INFO("Slot %d: saved %zu tokens to prompt cache (%.3f MiB)",
                 slot.id, tokens.size(),
                 (float)entry->size_bytes() / 1024.0f / 1024.0f);
}

ServerSlot *Llama::get_available_slot() {
  return this->slot_manager_->get_available_slot();
}

ServerSlot *Llama::wait_for_available_slot() {
  return this->slot_manager_->wait_for_available_slot();
}

ServerSlot *Llama::get_slot_by_id(int id) {
  return this->slot_manager_->get_slot_by_id(id);
}

ServerSlot *Llama::get_slot_by_gid(uint64_t gid) {
  return this->slot_manager_->get_slot_by_gid(gid);
}

bool Llama::process_token(ServerSlot *slot, CompletionOutput *result) {
  if (this->task_registry_->is_cancel_requested(slot->goal_id)) {
    slot->stop = CANCEL;
    slot->has_next_token = false;
    return false;
  }
  const std::string token_str = result->text_to_send;
  slot->sampled = result->token;

  slot->generated_text += token_str;
  slot->generated_tokens.push_back(result->token);
  slot->has_next_token = true;

  // check if there is incomplete UTF-8 character at the end
  bool incomplete = llama_utils::validate_utf8(slot->generated_text) <
                    slot->generated_text.size();

  // search stop word and delete it
  if (!incomplete) {
    size_t pos = std::min(slot->n_sent_text, slot->generated_text.size());

    const std::string str_test = slot->generated_text.substr(pos);
    bool send_text = true;

    size_t stop_pos =
        slot->find_stopping_strings(str_test, token_str.size(), true);
    if (stop_pos != std::string::npos) {
      slot->generated_text.erase(slot->generated_text.begin() + pos + stop_pos,
                                 slot->generated_text.end());
      pos = std::min(slot->n_sent_text, slot->generated_text.size());

    } else if (slot->has_next_token) {
      stop_pos = slot->find_stopping_strings(str_test, token_str.size(), false);
      send_text = (stop_pos == std::string::npos);
    }

    if (send_text) {
      result->text_to_send = slot->generated_text.substr(pos);
      slot->n_sent_text += result->text_to_send.size();
    } else {
      result->text_to_send.clear();
    }
  }

  if (incomplete) {
    // still waiting for the rest of a UTF-8 sequence, keep going
    slot->has_next_token = true;
  } else {
    LLAMA_LOG_DEBUG("Generated token: '%s'", result->text_to_send.c_str());
  }

  // if context shifting is disabled, make sure that we don't run out of context
  if (!this->params.ctx_shift && slot->n_past + 1 >= slot->n_ctx) {
    slot->stop = FULL_STOP;
    slot->has_next_token = false;

    LLAMA_LOG_INFO(
        "stopped due to running out of context, n_past = %d, n_ctx = %d\n",
        slot->n_past, slot->n_ctx);
  }

  // Check the limits (n_predict)
  if (slot->n_decoded > 0 && slot->has_next_token &&
      slot->params.n_predict != -1 &&
      slot->n_decoded >= slot->params.n_predict) {
    slot->stop = FULL_STOP;
    slot->has_next_token = false;

    LLAMA_LOG_INFO("stopped by limit, n_decoded = %d, n_predict = %d\n",
                   slot->n_decoded, slot->params.n_predict);
  }

  if (slot->has_new_line) {
    if (slot->params.n_indent > 0) {
      if (slot->last_nl_pos > 0) {
        size_t pos = slot->last_nl_pos;

        int n_indent = 0;
        while (pos < slot->generated_text.size() &&
               (slot->generated_text[pos] == ' ' ||
                slot->generated_text[pos] == '\t')) {
          n_indent++;
          pos++;
        }

        if (pos < slot->generated_text.size() &&
            n_indent < slot->params.n_indent) {
          slot->stop = FULL_STOP;
          slot->has_next_token = false;

          // cut the last line
          slot->generated_text.erase(pos, std::string::npos);

          LLAMA_LOG_INFO(
              "stopped by indentation limit, n_decoded = %d, n_indent = %d\n",
              slot->n_decoded, n_indent);
        }
      }

      // find the next new line
      {
        const size_t pos = slot->generated_text.find('\n', slot->last_nl_pos);

        if (pos != std::string::npos) {
          slot->last_nl_pos = pos + 1;
        }
      }
    }
  }

  // if context shift is disabled, we stop when it reaches the context limit
  if (!this->params.ctx_shift && slot->n_past >= slot->n_ctx) {
    slot->stop = FULL_STOP;
    slot->has_next_token = false;

    LLAMA_LOG_INFO(
        "stopped due to running out of context capacity, n_past = %d, "
        "n_prompt_tokens = %d, n_decoded = %d, n_ctx = %d\n",
        slot->n_past, slot->n_prompt_tokens, slot->n_decoded, slot->n_ctx);
  }

  if (llama_vocab_is_eog(this->get_vocab(), result->token)) {
    slot->stop = FULL_STOP;
    slot->has_next_token = false;
    slot->generated_text.erase(slot->generated_text.end() - token_str.size(),
                               slot->generated_text.end());
    if (!slot->generated_tokens.empty()) {
      slot->generated_tokens.pop_back();
    }

    LLAMA_LOG_INFO("%s", "stopped by EOS\n");
  } else if (slot->stream_callback && !result->text_to_send.empty()) {
    slot->stream_callback(*result, slot);
  }

  const auto n_ctx_train = llama_model_n_ctx_train(this->model);

  if (!this->params.ctx_shift && slot->n_predict < 1 &&
      slot->params.n_predict < 1 &&
      slot->n_prompt_tokens + slot->n_decoded >= n_ctx_train) {
    slot->stop = FULL_STOP;
    slot->has_next_token = false; // stop prediction

    LLAMA_LOG_WARN("stopped by context limit\n"
                   "n_predict (%d) is set for infinite generation. "
                   "Limiting generated tokens to n_ctx_train (%d)\n",
                   slot->params.n_predict, n_ctx_train);
  }

  auto n_remaining = slot->params.n_predict < 1
                         ? -1
                         : slot->params.n_predict - slot->n_decoded;

  LLAMA_LOG_DEBUG("n_decoded = %d, n_remaining = %d, next token: %5d '%s'\n",
                  slot->n_decoded, n_remaining, result->token,
                  token_str.c_str());

  return slot->has_next_token; // continue
}

std::vector<llama_token>
Llama::truncate_tokens(const std::vector<llama_token> &tokens, int limit_size,
                       bool add_eos) {

  std::vector<llama_token> new_tokens = tokens;

  // Reserve space for EOS token if needed
  int effective_limit = limit_size;
  if (add_eos && !tokens.empty() && tokens.back() != this->get_token_eos()) {
    effective_limit = limit_size - 1;
  }

  if ((int)tokens.size() > effective_limit) {
    LLAMA_LOG_WARN("Prompt too long %ld, limit size %d, truncating...",
                   tokens.size(), limit_size);
    new_tokens.resize(effective_limit);
  }

  // add eos if not present
  if (add_eos && !new_tokens.empty() &&
      new_tokens.back() != this->get_token_eos()) {
    new_tokens.push_back(this->get_token_eos());
  }

  return new_tokens;
}

/*
*****************************
*  SPECULATIVE DECODING     *
*****************************
*/
void Llama::init_speculative() {
  auto &spec_params = this->params.speculative;

  // Check if speculative decoding is configured
  bool has_draft = spec_params.has_dft();
  bool has_any_spec =
      std::any_of(spec_params.types.begin(), spec_params.types.end(),
                  [](auto t) { return t != COMMON_SPECULATIVE_TYPE_NONE; });

  if (!has_draft && !has_any_spec) {
    LLAMA_LOG_INFO("Speculative decoding not configured, skipping "
                   "initialization");
    return;
  }

  // Skip speculative for embedding/reranking models
  if (this->is_embedding() || this->is_reranking()) {
    LLAMA_LOG_WARN(
        "Speculative decoding is not supported with embedding/reranking "
        "models, skipping");
    return;
  }

  // Only supported with n_parallel=1 (single slot)
  if (this->params.n_parallel != 1) {
    LLAMA_LOG_WARN("Speculative decoding requires n_parallel=1, but got %d. "
                   "Skipping speculative initialization",
                   this->params.n_parallel);
    return;
  }

  // Check compatibility - speculative decoding requires seq_rm support
  if (common_context_can_seq_rm(this->ctx) == COMMON_CONTEXT_SEQ_RM_TYPE_NO) {
    LLAMA_LOG_WARN("Target context is not compatible with speculative "
                   "decoding (no sequence removal support), skipping");
    return;
  }

  const bool is_mtp = std::any_of(
      spec_params.types.begin(), spec_params.types.end(),
      [](auto t) { return t == COMMON_SPECULATIVE_TYPE_DRAFT_MTP; });

  if (has_draft) {
    // Start from a full copy of target params to inherit n_ctx, flash_attn,
    // and other settings; then override draft-specific fields.
    // Mirrors server-context.cpp draft model initialization.
    auto params_dft = this->params;
    params_dft.model = spec_params.draft.mparams;
    params_dft.n_gpu_layers = spec_params.draft.n_gpu_layers;
    params_dft.cache_type_k = spec_params.draft.cache_type_k;
    params_dft.cache_type_v = spec_params.draft.cache_type_v;
    if (spec_params.draft.cpuparams.n_threads > 0) {
      params_dft.cpuparams = spec_params.draft.cpuparams;
      params_dft.cpuparams_batch = spec_params.draft.cpuparams_batch;
    }

    LLAMA_LOG_INFO("Loading draft model: %s", params_dft.model.path.c_str());
    auto mparams_dft = common_model_params_to_llama(params_dft);
    this->model_dft_ =
        llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft);
    if (this->model_dft_ == nullptr) {
      LLAMA_LOG_ERROR("Failed to load draft model '%s', speculative decoding "
                      "will be disabled",
                      params_dft.model.path.c_str());
      return;
    }

    auto cparams_dft = common_context_params_to_llama(params_dft);
    if (is_mtp) {
      cparams_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }

    cparams_dft.n_rs_seq = 0;
    this->ctx_dft_ = llama_init_from_model(this->model_dft_, cparams_dft);
    if (this->ctx_dft_ == nullptr) {
      LLAMA_LOG_ERROR("Failed to create draft context, speculative decoding "
                      "will be disabled");
      llama_model_free(this->model_dft_);
      this->model_dft_ = nullptr;
      return;
    }

    if (spec_params.types[0] == COMMON_SPECULATIVE_TYPE_NONE) {
      spec_params.types = {COMMON_SPECULATIVE_TYPE_DRAFT_SIMPLE};
    }

  } else if (is_mtp) {
    // MTP heads are embedded in the main model GGUF; create a dedicated MTP
    // context from it (mirrors server-context.cpp).
    LLAMA_LOG_INFO("Creating MTP context from main model "
                   "(no separate draft model)");
    auto cparams_mtp = common_context_params_to_llama(this->params);
    cparams_mtp.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    cparams_mtp.n_rs_seq = 0;
    this->ctx_dft_ = llama_init_from_model(this->model, cparams_mtp);
    if (this->ctx_dft_ == nullptr) {
      LLAMA_LOG_ERROR("Failed to create MTP context, speculative decoding "
                      "will be disabled");
      return;
    }
  }

  if (this->ctx_dft_ != nullptr) {
    spec_params.draft.ctx_tgt = this->ctx;
    spec_params.draft.ctx_dft = this->ctx_dft_;
  }

  // Initialize speculative decoder
  this->speculative_ = common_speculative_init(spec_params, 1);

  if (this->speculative_ == nullptr) {
    LLAMA_LOG_ERROR("Failed to initialize speculative decoder");
    if (this->ctx_dft_ != nullptr) {
      llama_free(this->ctx_dft_);
      this->ctx_dft_ = nullptr;
    }
    if (this->model_dft_ != nullptr) {
      llama_model_free(this->model_dft_);
      this->model_dft_ = nullptr;
    }
    return;
  }

  LLAMA_LOG_INFO("Speculative decoding initialized (types: %s, n_max: %d, "
                 "n_min: %d, p_min: %.2f)",
                 common_speculative_all_types_str(), spec_params.draft.n_max,
                 spec_params.draft.n_min, spec_params.draft.p_min);
}

bool Llama::speculative_generation_step(ServerSlot *slot) {
  const auto &spec_params = this->params.speculative;

  // We need the prompt_tgt (all tokens processed so far, excluding the last
  // one) and id_last (the last token sampled).

  // Build prompt_tgt from the slot's prompt tokens (already KV-cached)
  // plus any generated tokens so far (excluding the most recent one which
  // is id_last).
  llama_tokens prompt_tgt;
  prompt_tgt.reserve(slot->prompt_tokens.size() +
                     slot->generated_tokens.size());

  // Add all prompt tokens
  for (auto token : slot->prompt_tokens) {
    if (token != LLAMA_TOKEN_NULL) {
      prompt_tgt.push_back(token);
    }
  }

  // Add all generated tokens except the last (which is id_last)
  if (!slot->generated_tokens.empty()) {
    for (size_t i = 0; i < slot->generated_tokens.size() - 1; ++i) {
      prompt_tgt.push_back(slot->generated_tokens[i]);
    }
  }

  llama_token id_last = slot->generated_tokens.empty()
                            ? slot->prompt_tokens.back()
                            : slot->generated_tokens.back();

  // Generate draft tokens
  llama_tokens draft;
  {
    auto &dparams =
        common_speculative_get_draft_params(this->speculative_, slot->id);
    dparams = {
        /* .drafting = */ true,
        /* .n_max    = */ -1,
        /* .pos0     = */ slot->n_past,
        /* .id_last  = */ id_last,
        /* .prompt   = */ &prompt_tgt,
        /* .result   = */ &draft,
    };
    common_speculative_draft(this->speculative_);
  }

  // Build batch: [id_last, draft0, draft1, ..., draftN-1]
  this->batch.clear();
  this->batch.add(id_last, slot->n_past, slot->id, true);

  // Skip small drafts
  if ((int)draft.size() < spec_params.draft.n_min) {
    draft.clear();
  }

  for (size_t i = 0; i < draft.size(); ++i) {
    this->batch.add(draft[i], slot->n_past + 1 + i, slot->id, true);
  }

  // Roll back ctx_dft after draft: draft() advanced ctx_dft's KV cache to
  // n_past + n_drafted. process() will decode the verify batch starting at
  // n_past, which M-RoPE rejects unless we first remove those positions.
  // Mirrors server-context.cpp: common_context_seq_rm(ctx_dft, id,
  // ckpt.pos_max+1, -1)
  if (this->ctx_dft_ != nullptr) {
    llama_memory_seq_rm(llama_get_memory(this->ctx_dft_), slot->id,
                        slot->n_past, -1);
  }

  // Decode the batch on the target model
  const int ret =
      llama_process(this->ctx, LLAMA_PROCESS_TYPE_DECODE, this->batch.get());
  if (ret != 0) {
    LLAMA_LOG_ERROR("Speculative decode failed with error %d (slot id=%d "
                    "gid=%lu n_past=%d n_ctx=%d n_decoded=%d)",
                    ret, slot->id, slot->goal_id, slot->n_past, slot->n_ctx,
                    slot->n_decoded);
    slot->stop = ABORT;
    slot->has_next_token = false;
    return false;
  }

  // Update hidden states in the speculative pipeline (required for MTP;
  // no-op for other types).
  common_speculative_process(this->speculative_, this->batch);

  // Verify draft tokens using the target sampler
  const auto ids =
      common_sampler_sample_and_accept_n(slot->sampler, this->ctx, draft);

  // ids always has at least 1 token (the one the target model would have
  // sampled) ids.size()-1 draft tokens were accepted

  const int n_accepted = (int)ids.size() - 1;
  common_speculative_accept(this->speculative_, slot->id, n_accepted);

  LLAMA_LOG_DEBUG("Speculative: drafted %d, accepted %d/%d", (int)draft.size(),
                  n_accepted, (int)draft.size());

  // Process accepted tokens + the final sampled token
  bool should_continue = true;

  for (size_t i = 0; i < ids.size(); ++i) {
    // Update prompt_tgt for future calls
    prompt_tgt.push_back(id_last);
    id_last = ids[i];

    // Advance n_past for each accepted/sampled token
    slot->n_past += 1;
    slot->n_kv_cache += 1;

    // Build CompletionOutput for this token
    CompletionOutput result;
    result.token = id_last;
    result.text_to_send = common_token_to_piece(this->ctx, id_last);
    result.probs = this->get_probs(slot);
    slot->generated_probs.push_back(result.probs);

    slot->n_decoded += 1;

    // Run process_token to handle stop words, limits, EOG, etc.
    if (!this->process_token(slot, &result)) {
      should_continue = false;
      break;
    }
  }

  // Clear KV cache for any extra draft tokens that were rejected on the
  // target context, and mirror the trim on the MTP draft context.
  llama_memory_seq_rm(llama_get_memory(this->ctx), slot->id, slot->n_past, -1);
  if (this->ctx_dft_ != nullptr) {
    llama_memory_seq_rm(llama_get_memory(this->ctx_dft_), slot->id,
                        slot->n_past, -1);
  }

  if (!should_continue) {
    this->send_completion_result(slot);
    this->release_slot(slot);
  }

  return should_continue;
}

void Llama::run_loop() {
  while (!this->canceled) {
    // Only the inference thread changes slot state. RESERVED slots are still
    // being populated; their cancellation remains pending until publication.
    for (auto &slot : this->server_slots) {
      if (slot.state != SLOT_STATE_IDLE && slot.state != SLOT_STATE_RESERVED &&
          this->task_registry_->is_cancel_requested(slot.goal_id)) {
        slot.stop = CANCEL;
        LLAMA_LOG_INFO("Canceled slot %d, committed KV=%d", slot.id,
                       slot.n_kv_cache);
        this->send_completion_result(&slot);
        this->release_slot(&slot);
      }
    }

    // Check if any slots are being processed
    bool any_processing = false;
    for (auto &slot : this->server_slots) {
      if (slot.is_processing()) {
        any_processing = true;
        break;
      }
    }

    if (!any_processing) {
      // No slots are being processed, we can sleep
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      LLAMA_LOG_DEBUG("No active slots, sleeping...");
      continue;
    }

    // Apply context shift
    if (this->params.ctx_shift) {
      for (auto &slot : this->server_slots) {
        if (slot.state != SLOT_STATE_GENERATING) {
          continue;
        }

        // keep a pre-shift snapshot so the prefix stays recoverable
        this->maybe_create_checkpoint(slot);

        // Classic sliding window
        if (this->params.grp_attn_n <= 1) {
          if (slot.n_past + 1 > slot.n_ctx) {
            const int n_keep = this->params.n_keep;

            const int n_left = slot.n_past - n_keep;
            if (n_left <= 0) {
              continue;
            }

            const int n_discard = n_left / 2;

            llama_memory_seq_rm(this->get_memory(), slot.id, n_keep,
                                n_keep + n_discard);
            llama_memory_seq_add(this->get_memory(), slot.id,
                                 n_keep + n_discard, slot.n_past, -n_discard);

            slot.n_past -= n_discard;
            // Sliding window shifted KV positions: drop the cached prefix.
            slot.invalidate_kv_cache();
          }

        } else {
          // Self-Extend
          const int ga_n = this->params.grp_attn_n;
          const int ga_w = this->params.grp_attn_w;

          while (slot.n_past >= slot.ga_i + ga_w) {
            const int ib = (ga_n * slot.ga_i) / ga_w;
            const int bd = (ga_w / ga_n) * (ga_n - 1);
            const int dd = (ga_w / ga_n) - ib * bd - ga_w;

            llama_memory_seq_add(this->get_memory(), slot.id, slot.ga_i,
                                 slot.n_past, ib * bd);

            llama_memory_seq_div(this->get_memory(), slot.id,
                                 slot.ga_i + ib * bd,
                                 slot.ga_i + ib * bd + ga_w, ga_n);

            llama_memory_seq_add(this->get_memory(), slot.id,
                                 slot.ga_i + ib * bd + ga_w,
                                 slot.n_past + ib * bd, dd);

            slot.n_past -= bd;
            // Self-Extend reshuffled KV positions: drop the cached prefix.
            slot.invalidate_kv_cache();

            slot.ga_i += ga_w / ga_n;
          }
        }
      }
    }

    ServerSlot *slot_batched = nullptr;

    // ====================================================================
    // Speculative decoding path: process generating slots with speculation
    // ====================================================================
    if (this->is_speculative()) {
      bool handled_speculative = false;
      for (auto &slot : this->server_slots) {
        if (slot.state == SLOT_STATE_GENERATING &&
            slot.task_type == SERVER_TASK_TYPE_COMPLETION) {
          this->speculative_generation_step(&slot);
          handled_speculative = true;
        }
      }
      // If we handled speculative generation, skip normal generation batch
      // but still need to handle prompt processing below
      if (handled_speculative) {
        // Check if there are any prompt-processing slots that still need work
        bool has_prompt_work = false;
        for (auto &slot : this->server_slots) {
          if (slot.state == SLOT_STATE_PROCESSING_PROMPT ||
              slot.state == SLOT_STATE_STARTED) {
            has_prompt_work = true;
            break;
          }
        }
        if (!has_prompt_work) {
          continue; // all generating slots handled by speculation
        }
      }
    }

    // start populating the batch for this iteration
    this->batch.clear();

    for (auto &slot : this->server_slots) {
      if (slot.state != SLOT_STATE_GENERATING) {
        continue;
      }

      // Skip generating slots that are handled by speculative decoding
      if (this->is_speculative() &&
          slot.task_type == SERVER_TASK_TYPE_COMPLETION) {
        continue;
      }

      if (!slot_batched) {
        slot_batched = &slot;
      }

      slot.i_batch = this->batch.size();
      this->batch.add(slot.sampled, slot.n_past, slot.id, true);

      slot.n_past += 1;
    }

    // Process prompts (new inputs)
    int32_t n_batch = llama_n_batch(this->ctx);
    if (this->params.cont_batching || this->batch.size() == 0) {
      for (auto &slot : this->server_slots) {
        // ensure batch-compatibility across slots
        if (slot.is_processing()) {
          if (!slot_batched) {
            slot_batched = &slot;
          }
        }

        // only handle newly started or actively processing prompt slots
        if (slot.state != SLOT_STATE_PROCESSING_PROMPT &&
            slot.state != SLOT_STATE_STARTED) {
          continue;
        }

        auto &prompt_tokens = slot.prompt_tokens;

        // first-time setup for a new prompt
        if (slot.state == SLOT_STATE_STARTED) {

          slot.n_prompt_tokens = prompt_tokens.size();
          slot.state = SLOT_STATE_PROCESSING_PROMPT;

          // empty prompt -> release and send empty response
          if (prompt_tokens.empty()) {
            LLAMA_LOG_WARN("Empty prompt on slot %d", slot.id);
            this->fail_pending(slot.goal_id, "Empty prompt");
            this->release_slot(&slot);
            continue;
          }

          if (slot.n_prompt_tokens > slot.n_ctx) {
            LLAMA_LOG_WARN("Prompt exceeds context size for slot %d", slot.id);
            this->fail_pending(slot.goal_id, "Prompt exceeds context size");
            this->release_slot(&slot);
            continue;
          }

          if (slot.task_type == SERVER_TASK_TYPE_DECISION &&
              this->decision_model_ != nullptr &&
              this->decision_model_->is_joint() &&
              !slot.map_pos_to_media.empty() &&
              (size_t)slot.n_prompt_tokens >
                  (size_t)std::min(llama_n_batch(this->ctx),
                                   llama_n_ubatch(this->ctx))) {
            LLAMA_LOG_WARN("Decision prompt exceeds the batch size on slot %d",
                           slot.id);
            this->fail_pending(slot.goal_id,
                               "The question, its options and the state must "
                               "fit in one batch; increase context.n_batch and "
                               "context.n_ubatch");
            this->release_slot(&slot);
            continue;
          }

          // Reuse the KV-cached prefix when allowed; fall back to a full wipe
          // when the prompt diverges or caching is disabled for this slot.
          const bool caching_allowed = this->params.cache_prompt &&
                                       slot.cache_prompt &&
                                       !this->is_speculative();
          auto *mem = llama_get_memory(this->ctx);

          size_t reused = 0;
          if (caching_allowed) {
            const bool can_chunk_reuse =
                this->params.n_cache_reuse > 0 && slot.kv_positions_valid &&
                slot.map_pos_to_media.empty() && llama_memory_can_shift(mem);
            reused = can_chunk_reuse
                         ? slot.reuse_kv_chunks(mem, prompt_tokens,
                                                this->params.n_cache_reuse)
                         : slot.find_reusable_prefix(prompt_tokens);
          }

          if (reused == 0 && caching_allowed && slot.kv_positions_valid &&
              slot.map_pos_to_media.empty() &&
              this->params.mmproj.path.empty() &&
              !llama_model_is_recurrent(this->model) &&
              !llama_model_is_hybrid(this->model) &&
              this->partial_seq_removal_ &&
              llama_model_n_swa(this->model) == 0) {
            reused = std::min(slot.common_prefix_len(prompt_tokens),
                              static_cast<size_t>(slot.n_kv_cache));
            if (reused == prompt_tokens.size() && reused > 0) {
              --reused; // obtain fresh logits for a subsequent completion
            }
          }

          if (reused == 0 && caching_allowed) {
            this->try_restore_checkpoint(slot, prompt_tokens, reused);
            if (reused == 0) {
              this->try_load_prompt_cache(slot, prompt_tokens, reused);
            }
          }

          if (reused > 0) {
            llama_memory_seq_rm(mem, slot.id, static_cast<llama_pos>(reused),
                                -1);
            slot.n_past = static_cast<int32_t>(reused);
            LLAMA_LOG_INFO("Slot %d: reused %zu/%zu tokens from KV cache",
                           slot.id, reused, prompt_tokens.size());
          } else {
            slot.n_past = 0;
            llama_memory_seq_rm(mem, slot.id, -1, -1);
            slot.kv_positions_valid = true;
            slot.checkpoints.clear();
            LLAMA_LOG_INFO("Slot %d: KV cache cold start (0/%zu reused)",
                           slot.id, prompt_tokens.size());
          }

          // clear idle slots to free up VRAM when a new task starts
          if (this->params.cache_idle_slots) {
            for (auto &other : this->server_slots) {
              if (!other.is_processing() && other.id != slot.id &&
                  other.n_past > 0) {
                LLAMA_LOG_INFO("Clearing idle slot %d (n_past=%d) for new task",
                               other.id, other.n_past);
                this->save_prompt_to_cache(other);
                llama_memory_seq_rm(mem, other.id, -1, -1);
                other.n_past = 0;
                other.prompt_tokens.clear();
                other.invalidate_kv_cache();
              }
            }
          }

          // ensure at least one token will be evaluated
          if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
            slot.n_past--;
          }

          slot.n_kv_cache = slot.n_past;
          slot.kv_cached_tokens.assign(prompt_tokens.begin(),
                                       prompt_tokens.begin() + slot.n_past);
          slot.n_prompt_tokens_processed = 0;
        }

        // skip if batch is already full
        if (static_cast<uint32_t>(this->batch.size()) >=
            llama_n_batch(this->ctx)) {
          continue;
        }

        const bool joint_decision =
            slot.task_type == SERVER_TASK_TYPE_DECISION &&
            this->decision_model_ != nullptr &&
            this->decision_model_->is_joint();

        // process MTMD chunks if present; joint decision media is handled
        // inline by the enqueue loop below so the whole prompt is decoded
        // in one batch (the Clef head is stateless and cannot be split)
        if (!joint_decision && slot.n_past < slot.n_prompt_tokens &&
            slot.prompt_tokens[slot.n_past] == LLAMA_TOKEN_NULL) {
          process_mtmd_chunk(&slot);
        }

        // enqueue prompt tokens up to the available batch capacity
        bool enqueue_failed = false;
        while (slot.n_past < slot.n_prompt_tokens) {
          if (static_cast<uint32_t>(this->batch.size()) >=
              llama_n_batch(this->ctx)) {
            break; // batch is full, continue in the next iteration
          }

          llama_token cur_tok = slot.prompt_tokens[slot.n_past];
          if (cur_tok == LLAMA_TOKEN_NULL) {
            if (!joint_decision) {
              break; // end of text chunk
            }

            // a joint decision prompt must own its batch: the Clef head
            // reads decision_order spans and requires a single sequence
            bool foreign = false;
            for (const auto &batch_token : this->batch.tokens) {
              if (batch_token.seq_id != slot.id ||
                  !batch_token.seq_ids_extra.empty()) {
                foreign = true;
                break;
              }
            }
            if (foreign) {
              LLAMA_LOG_WARN(
                  "Joint decision prompt cannot share a batch on slot %d",
                  slot.id);
              this->fail_pending(slot.goal_id,
                                 "Joint decision prompts cannot share a "
                                 "batch; set context.n_parallel to 1");
              this->release_slot(&slot);
              enqueue_failed = true;
              break;
            }

            bool ok = false;
            try {
              ok = this->process_decision_mtmd_batch(&slot);
            } catch (const std::exception &e) {
              LLAMA_LOG_ERROR("Failed to process decision media: %s", e.what());
              ok = false;
            }
            if (!ok) {
              this->fail_pending(slot.goal_id,
                                 "Failed to process decision media");
              this->release_slot(&slot);
              enqueue_failed = true;
              break;
            }
            continue;
          }

          const bool need_embd = this->is_embedding() || this->is_reranking() ||
                                 (this->is_decision() &&
                                  this->decision_model_->needs_embeddings());

          this->batch.add(cur_tok, slot.n_past, slot.id, need_embd);

          if (!slot.decision.order.empty() && slot.n_past >= 0 &&
              (size_t)slot.n_past < slot.decision.order.size()) {
            this->batch.tokens.back().decision_order =
                slot.decision.order[slot.n_past];
          }

          slot.n_prompt_tokens_processed++;
          slot.n_past++;
        }

        if (enqueue_failed) {
          continue;
        }

        if (joint_decision && slot.n_past < slot.n_prompt_tokens) {
          // the batch capacity was reached before the whole prompt was
          // enqueued; a split prompt would be discarded by the stateless
          // Clef encoder
          LLAMA_LOG_WARN(
              "Joint decision prompt does not fit the batch on slot %d",
              slot.id);
          this->fail_pending(slot.goal_id,
                             "The question, its options and the state must "
                             "fit in one batch; increase context.n_batch and "
                             "context.n_ubatch");
          this->release_slot(&slot);
          continue;
        }

        LLAMA_LOG_INFO("Processed %d/%d prompt tokens for slot %d",
                       slot.n_prompt_tokens_processed, slot.n_prompt_tokens,
                       slot.id);

        if (slot.n_past == slot.n_prompt_tokens) {
          slot.state = SLOT_STATE_DONE_PROMPT;

          // reset sampler and virtually accept the prompt so the next token can
          // be sampled
          common_sampler_reset(slot.sampler);
          for (int i = 0; i < slot.n_prompt_tokens; ++i) {
            llama_token id = slot.prompt_tokens[i];
            if (id != LLAMA_TOKEN_NULL) {
              common_sampler_accept(slot.sampler, id, false);
            }
          }

          // request logits for the last prompt token
          this->batch.set_output(this->batch.size() - 1, true);

          slot.n_decoded = 0;
          slot.i_batch = this->batch.size() - 1;

          // Only completion tasks benefit from prefix reuse; an embedding or
          // rerank prompt would mismatch the next request's task type.
          // Cache metadata is committed only after successful decode below.

          LLAMA_LOG_INFO("prompt done, n_past = %d, n_tokens = %d\n",
                         slot.n_past, this->batch.size());
        }

        if (static_cast<uint32_t>(this->batch.size()) >=
            llama_n_batch(this->ctx)) {
          LLAMA_LOG_DEBUG("Batch full, remaining prompt tokens will be "
                          "processed in the next iteration");
          continue;
        }
      }
    }

    // Check if there are no tokens to decode (expected when slots are
    // SLOT_STATE_RESERVED — waiting for prompt population by the worker thread)
    if (this->batch.size() == 0) {
      LLAMA_LOG_DEBUG("No tokens to decode in this iteration (slot may be "
                      "reserved, waiting for prompt)");
      continue;
    }

    int32_t i_next = 0;

    LLAMA_LOG_DEBUG("Decoding batch of %d tokens", this->batch.size());
    for (int32_t i = 0; i < this->batch.size(); i = i_next) {
      const int32_t n_tokens = std::min(n_batch, this->batch.size() - i);

      llama_batch_ext *batch_view = this->batch.get_sub_batch(i, n_tokens);

      // A shared batch must never be aborted for just one of its goals.
      // Other configurations retain cooperative cancellation at decode/step
      // boundaries; the CPU callback is used only with rollback-safe text KV.
      ServerSlot *abort_slot = nullptr;
      if (this->supports_precompute() && this->params.n_gpu_layers == 0 &&
          this->server_slots[0].map_pos_to_media.empty() &&
          this->server_slots[0].kv_positions_valid) {
        abort_slot = &this->server_slots[0];
      }
      std::function<bool()> abort_callback;
      if (abort_slot != nullptr) {
        const uint64_t gid = abort_slot->goal_id;
        abort_callback = [this, gid] {
          return this->task_registry_->is_cancel_requested(gid);
        };
      }
      int ret;
      {
        ScopedDecodeAbort abort_guard(this->ctx, std::move(abort_callback));
        ret = llama_process(this->ctx, LLAMA_PROCESS_TYPE_DECODE, batch_view);
      }
      if (ret == 2) {
        // v0.4.1 removes the failed ubatch and all following positions.
        // Earlier completed ubatches remain materialized in text attention KV.
        if (abort_slot != nullptr) {
          auto &slot = *abort_slot;
          const int committed = std::max(
              0,
              static_cast<int>(
                  llama_memory_seq_pos_max(this->get_memory(), slot.id) + 1));
          if (committed > slot.n_past ||
              (committed > 0 &&
               llama_memory_seq_pos_min(this->get_memory(), slot.id) != 0)) {
            slot.invalidate_kv_cache();
            this->fail_pending(slot.goal_id,
                               "Aborted decode returned noncontiguous KV");
            this->release_slot(&slot);
            break;
          }
          slot.n_prompt_tokens_processed = std::max(
              0, slot.n_prompt_tokens_processed - (slot.n_past - committed));
          slot.n_past = committed;
          slot.n_kv_cache = committed;
          slot.kv_cached_tokens.assign(
              slot.prompt_tokens.begin(),
              slot.prompt_tokens.begin() +
                  std::min(committed, slot.n_prompt_tokens));
          slot.checkpoints.clear();
          slot.stop = CANCEL;
          LLAMA_LOG_INFO("Canceled slot %d during decode: cached=%d/%d",
                         slot.id, committed, slot.n_prompt_tokens);
          this->send_completion_result(&slot);
          this->release_slot(&slot);
        } else {
          // A foreign/backend abort is not KV pressure and must not retry.
          for (auto &slot : this->server_slots) {
            if (slot.is_processing() && slot.state != SLOT_STATE_RESERVED) {
              slot.invalidate_kv_cache();
              this->fail_pending(slot.goal_id, "Unexpected decode abort");
              this->release_slot(&slot);
            }
          }
        }
        break;
      }

      if (ret != 0) {
        // Map common error cases to readable messages
        std::string err;
        if (n_batch == 1 && ret == 1) {
          err = "Context size has been exceeded.";
        } else if (ret == -1) {
          err = "Invalid input batch.";
        } else if (ret < -1) {
          err = "Compute error.";
        }

        if (!err.empty()) {
          LLAMA_LOG_ERROR("Decoding error: %s (ret=%d, n_batch=%d, i=%d, "
                          "batch_n_tokens=%d)",
                          err.c_str(), ret, n_tokens, i, this->batch.size());
          // A decode failure leaves the KV state of every active slot
          // unknown; drop their prefix caches so the next request re-processes.
          for (auto &slot : this->server_slots) {
            if (slot.is_processing()) {
              LLAMA_LOG_ERROR(
                  "  Active slot id=%d gid=%lu state=%d task_type=%d "
                  "n_past=%d n_ctx=%d n_prompt_tokens=%d n_decoded=%d",
                  slot.id, slot.goal_id, (int)slot.state, (int)slot.task_type,
                  slot.n_past, slot.n_ctx, slot.n_prompt_tokens,
                  slot.n_decoded);
            }
            slot.invalidate_kv_cache();
          }
          this->cancel();
          break; // abort the decode loop
        }

        // No readable error - likely KV pressure: backoff and retry smaller
        // batch window
        n_batch = std::max(1, n_batch / 2);
        continue; // retry current window with smaller n_batch
      }

      for (int32_t t = 0; t < n_tokens; ++t) {
        const auto &batch_token = this->batch.tokens[i + t];
        auto update_kv_cache = [&](llama_seq_id seq_id) {
          auto &slot = this->server_slots[seq_id];
          slot.n_kv_cache = std::max(slot.n_kv_cache, batch_token.pos[0] + 1);
        };
        update_kv_cache(batch_token.seq_id);
        for (const auto seq_id : batch_token.seq_ids_extra) {
          update_kv_cache(seq_id);
        }
      }
      for (auto &slot : this->server_slots) {
        if (slot.task_type == SERVER_TASK_TYPE_COMPLETION &&
            (slot.state == SLOT_STATE_PROCESSING_PROMPT ||
             slot.state == SLOT_STATE_DONE_PROMPT)) {
          slot.kv_cached_tokens.assign(
              slot.prompt_tokens.begin(),
              slot.prompt_tokens.begin() +
                  std::min(slot.n_kv_cache, slot.n_prompt_tokens));
        }
        if (slot.precompute && slot.is_processing()) {
          LLAMA_LOG_INFO("Precompute progress: cached=%d/%d", slot.n_kv_cache,
                         slot.n_prompt_tokens);
        }
      }
      i_next = i + n_tokens;
      n_batch = llama_n_batch(this->ctx);

      // Feed hidden states to speculative decoder (required for MTP).
      if (this->speculative_ != nullptr) {
        common_batch speculative_batch(this->ctx);
        for (int32_t t = 0; t < n_tokens; ++t) {
          const auto &batch_token = this->batch.tokens[i + t];
          speculative_batch.add(batch_token.id, batch_token.pos[0],
                                batch_token.seq_id, batch_token.output);
          for (const auto seq_id : batch_token.seq_ids_extra) {
            speculative_batch.add_seq(speculative_batch.size() - 1, seq_id);
          }
        }
        common_speculative_process(this->speculative_, speculative_batch);
      }

      // Consume results per-slot for the tokens we just decoded
      for (auto &slot : this->server_slots) {
        if (slot.state == SLOT_STATE_IDLE ||
            slot.state == SLOT_STATE_RESERVED || slot.i_batch < (int)i ||
            slot.i_batch >= (int)(i + n_tokens)) {
          continue;
        }

        if (this->task_registry_->is_cancel_requested(slot.goal_id)) {
          slot.stop = CANCEL;
          this->send_completion_result(&slot);
          this->release_slot(&slot);
          continue;
        }

        // snapshot unrollbackable state (SWA/recurrent) while it is valid
        this->maybe_create_checkpoint(slot);

        // If we just finished prompt eval for this slot, branch by task type
        if (slot.state == SLOT_STATE_DONE_PROMPT) {
          if (slot.task_type == SERVER_TASK_TYPE_EMBEDDING) {
            this->send_embedding_result(&slot, i, n_tokens);
            this->release_slot(&slot);
            slot.i_batch = -1;
            continue;
          }

          if (slot.task_type == SERVER_TASK_TYPE_RERANK) {
            this->send_rerank_result(&slot, i, n_tokens);
            this->release_slot(&slot);
            slot.i_batch = -1;
            continue;
          }

          if (slot.task_type == SERVER_TASK_TYPE_DECISION) {
            this->send_decision_result(&slot, i, n_tokens);
            this->release_slot(&slot);
            slot.i_batch = -1;
            continue;
          }

          if (slot.precompute) {
            LLAMA_LOG_INFO("Precompute complete: cached=%d/%d", slot.n_kv_cache,
                           slot.n_prompt_tokens);
            slot.stop = FULL_STOP;
            this->send_completion_result(&slot);
            this->release_slot(&slot);
            continue;
          }

          // Default path: continue into text generation
          slot.state = SLOT_STATE_GENERATING;

          // If speculative decoding is enabled, sample the first token here
          // (id_last) and then defer to the speculative path on next iteration.
          if (this->is_speculative() &&
              slot.task_type == SERVER_TASK_TYPE_COMPLETION) {
            const int tok_idx = slot.i_batch - i;
            llama_token id =
                common_sampler_sample(slot.sampler, this->ctx, tok_idx);
            slot.i_batch = -1;

            common_sampler_accept(slot.sampler, id, true);
            slot.n_decoded += 1;

            // Initialize the speculative decoder with the prompt
            llama_tokens prompt_tgt;
            prompt_tgt.reserve(slot.prompt_tokens.size());
            for (auto token : slot.prompt_tokens) {
              if (token != LLAMA_TOKEN_NULL) {
                prompt_tgt.push_back(token);
              }
            }
            common_speculative_begin(this->speculative_, slot.id, prompt_tgt);

            CompletionOutput result;
            result.token = id;
            result.text_to_send = common_token_to_piece(this->ctx, id);
            result.probs = this->get_probs(&slot);
            slot.generated_probs.push_back(result.probs);

            if (!this->process_token(&slot, &result)) {
              this->send_completion_result(&slot);
              this->release_slot(&slot);
            }
            continue;
          }

        } else if (slot.state != SLOT_STATE_GENERATING) {
          continue;
        }

        // Index of this slot's token within the current decode window
        const int tok_idx = slot.i_batch - i;

        // Sample next token and advance sampler state
        llama_token id =
            common_sampler_sample(slot.sampler, this->ctx, tok_idx);
        slot.i_batch = -1;

        common_sampler_accept(slot.sampler, id, true);
        slot.n_decoded += 1;

        // Prepare token output
        CompletionOutput result;
        result.token = id;
        result.text_to_send = common_token_to_piece(this->ctx, id);
        result.probs = this->get_probs(&slot);
        slot.generated_probs.push_back(result.probs);

        // Stream token / check stopping conditions
        if (!this->process_token(&slot, &result)) {
          this->send_completion_result(&slot);
          this->release_slot(&slot);
          continue;
        }
      }
    }
  }

  LLAMA_LOG_INFO("Exiting run loop");
}

bool llama_ros::Llama::process_mtmd_chunk(llama_ros::ServerSlot *slot) {
  (void)slot;
  return false;
}

/*
*****************************
*   ASYNC TASK MANAGEMENT    *
*****************************
*/
std::future<ServerTaskResultPtr> Llama::register_pending(uint64_t goal_id) {
  return this->task_registry_->register_pending(goal_id);
}

void Llama::fulfill_pending(uint64_t goal_id, ServerTaskResultPtr r) {
  this->task_registry_->fulfill_pending(goal_id, std::move(r));
}

void Llama::fail_pending(uint64_t goal_id, std::string err) {
  this->task_registry_->fail_pending(goal_id, err);
}

/*
*****************************
*   REQUEST HANDLERS        *
*****************************
*/
void Llama::handle_embeddings_req(const std::string &input_prompt,
                                  ServerSlot *slot) {
  this->embedding_handler_->handle(input_prompt, slot);
}

void Llama::handle_rerank_req(const std::string &query,
                              const std::string &document, ServerSlot *slot) {
  this->rerank_handler_->handle(query, document, slot);
}

void Llama::handle_completion_req(const std::string &input_prompt,
                                  ServerSlot *slot,
                                  common_params_sampling sparams,
                                  ServerSlot::GenerateResponseCallback callback,
                                  std::vector<std::string> stop, bool reset) {
  this->completion_handler_->handle(input_prompt, slot, sparams, callback, stop,
                                    reset);
}

void Llama::handle_chat_completion_req(
    llama_utils::ChatCompletionsContext chat_context, ServerSlot *slot,
    ServerSlot::GenerateResponseCallback callback) {
  this->chat_completion_handler_->handle(chat_context, slot, callback);
}

/*
*****************************
*   RESULT HANDLERS         *
*****************************
*/
void Llama::send_embedding_result(ServerSlot *slot, int32_t off,
                                  int32_t n_tokens) {
  auto result = std::make_unique<ServerTaskResultEmbedding>();
  result->id_slot = slot->id;
  result->id = slot->goal_id;
  result->n_tokens = n_tokens;
  const int n_embd = llama_model_n_embd(this->model);

  std::vector<float> embd_res(n_embd, 0.0f);

  for (int32_t i = 0; i < n_tokens; ++i) {
    const auto &batch_token = this->batch.tokens[off + i];
    if (!batch_token.output || batch_token.seq_id != slot->id) {
      continue;
    }

    const float *embd = nullptr;
    if (llama_pooling_type(this->ctx) == LLAMA_POOLING_TYPE_NONE) {
      embd = llama_get_embeddings_ith(this->ctx, i);
    } else {
      embd = llama_get_embeddings_seq(this->ctx, batch_token.seq_id);
    }

    if (embd == nullptr) {
      LLAMA_LOG_ERROR("failed to get embeddings, token = %d, seq_id = %d\n",
                      batch_token.id, batch_token.seq_id);

      result->embeddings.push_back(std::vector<float>(n_embd, 0.0f));
      continue;
    }

    // normalize only when there is pooling
    if (llama_pooling_type(this->ctx) != LLAMA_POOLING_TYPE_NONE) {
      common_embd_normalize(embd, embd_res.data(), n_embd, 2);
      result->embeddings.push_back(embd_res);
      break;
    } else {
      result->embeddings.emplace_back(embd, embd + n_embd);
    }
  }

  const auto id = result->id;

  this->fulfill_pending(id, std::move(result));
}

void Llama::send_rerank_result(ServerSlot *slot, int32_t off,
                               int32_t n_tokens) {
  auto result = std::make_unique<ServerTaskResultRerank>();
  result->id_slot = slot->id;
  result->id = slot->goal_id;
  for (int32_t i = 0; i < n_tokens; ++i) {
    const auto &batch_token = this->batch.tokens[off + i];
    if (!batch_token.output || batch_token.seq_id != slot->id) {
      continue;
    }

    const float *embd = llama_get_embeddings_seq(this->ctx, batch_token.seq_id);
    if (embd == NULL) {
      embd = llama_get_embeddings_ith(this->ctx, i);
    }

    if (embd == NULL) {
      LLAMA_LOG_ERROR("failed to get embeddings, token = %d, seq_id = %d\n",
                      batch_token.id, batch_token.seq_id);

      result->score = -1e6;
      continue;
    }

    result->score = embd[0];
  }

  LLAMA_LOG_INFO("Rerank score: %f", result->score);
  const auto id = result->id;
  this->fulfill_pending(id, std::move(result));
}

void Llama::send_completion_result(ServerSlot *slot) {
  auto task_result = std::make_unique<ServerTaskResultCompletion>();
  task_result->id_slot = slot->id;
  task_result->id = slot->goal_id;

  task_result->content = slot->generated_text;
  task_result->tokens = {slot->generated_tokens};
  task_result->stop = slot->stop;
  task_result->prompt = this->detokenize(slot->prompt_tokens);
  task_result->stream = slot->stream;

  LLAMA_LOG_INFO("size logprobs: %lu for slot %d", slot->generated_probs.size(),
                 slot->id);
  task_result->probs_output = this->convert_probs_to_logprobs(slot);
  LLAMA_LOG_INFO("Length probs_output: %lu", task_result->probs_output.size());

  task_result->build_info =
      "b" + std::to_string(llama_build_number()) + "-" + llama_commit();
  task_result->oaicompat_model = this->get_metadata().general.name;
  task_result->oaicompat_cmpl_id = llama_utils::gen_chatcmplid();
  task_result->n_decoded = slot->n_decoded;
  task_result->n_prompt_tokens = slot->n_prompt_tokens;
  task_result->oaicompat_msg =
      slot->update_chat_msg(task_result->oaicompat_msg_diffs);

  // save the sequence state before the caller can reset/clear the context
  this->save_prompt_to_cache(*slot);

  const auto id = task_result->id;
  this->fulfill_pending(id, std::move(task_result));
}
