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

#include <cassert>
#include <chat.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <llama.h>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

#include "common.h"
#include "llama_utils/chat_utils.hpp"
#include "sampling.h"
#include "speculative.h"

#include "llama_ros/llama.hpp"
#include "llama_utils/logs.hpp"

using namespace llama_ros;

Llama::Llama(const common_params &params, std::string system_prompt,
             bool initial_reset)
    : params(params), system_prompt(system_prompt) {

  this->llama_init = common_init_from_params(this->params);

  print_build_info();

  // load model
  llama_backend_init();
  llama_numa_init(this->params.numa);

  this->model = this->llama_init->model();
  this->ctx = this->llama_init->context();
  this->lora_adapters = this->params.lora_adapters;

  if (this->model == NULL) {
    LLAMA_LOG_ERROR("Unable to load model");
    throw std::runtime_error("Unable to load model");
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
    slot.params.sampling = this->params.sampling;
    slot.params.n_keep = this->params.n_keep;
    slot.sampler = nullptr;

    slot.reset();
    this->server_slots.push_back(std::move(slot));
  }

  LLAMA_LOG_INFO("Initializing batch context");

  const int32_t n_batch = llama_n_batch(this->ctx);
  this->batch =
      llama_batch_init(std::max(n_batch, this->params.n_parallel), 0, 1);

  LLAMA_LOG_INFO("Model loaded successfully");

  // Initialize managers and handlers
  this->slot_manager_ = std::make_unique<SlotManager>(this->server_slots);
  this->task_registry_ = std::make_unique<TaskRegistry>();
  LLAMA_LOG_INFO("Initialized Slot Manager and Task Registry");

  this->embedding_handler_ = std::make_unique<EmbeddingRequestHandler>(this);
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

  // init threadpool
  LLAMA_LOG_INFO("llama threadpool init = n_threads = %d",
                 this->params.cpuparams.n_threads);

  ggml_threadpool_params tpp_batch =
      ggml_threadpool_params_from_cpu_params(this->params.cpuparams_batch);
  ggml_threadpool_params tpp =
      ggml_threadpool_params_from_cpu_params(this->params.cpuparams);

  set_process_priority(this->params.cpuparams.priority);

  LLAMA_LOG_INFO("loaded threadpool params");

  this->threadpool_batch = NULL;
  if (!ggml_threadpool_params_match(&tpp, &tpp_batch)) {
    this->threadpool_batch = ggml_threadpool_new(&tpp_batch);
    if (!this->threadpool_batch) {
      LLAMA_LOG_ERROR("Failed to create batch threadpool: n_threads %d",
                      tpp_batch.n_threads);
      return;
    }

    // start the non-batch threadpool in the paused state
    tpp.paused = true;
  }

  LLAMA_LOG_INFO("creating threadpool");

  this->threadpool = ggml_threadpool_new(&tpp);
  if (!this->threadpool) {
    LLAMA_LOG_ERROR("Failed to create threadpool: n_threads %d", tpp.n_threads);
    return;
  }

  LLAMA_LOG_INFO("attaching threadpool");
  llama_attach_threadpool(this->ctx, this->threadpool, this->threadpool_batch);

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
  LLAMA_LOG_INFO("llama.cpp: build = %d, commit = %s", LLAMA_BUILD_NUMBER,
                 LLAMA_COMMIT);
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

  llama_set_embeddings(this->ctx, this->is_embedding() || this->is_reranking());

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
  llama_backend_free();

  ggml_threadpool_free(this->threadpool);
  this->threadpool = nullptr;

  ggml_threadpool_free(this->threadpool_batch);
  this->threadpool_batch = nullptr;
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
  }

  llama_memory_clear(this->get_memory(), true);

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
  if (res >= 0) {
    metada_str = std::string(buffer.data(), buffer.size());
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

int Llama::get_int_metadata(const std::string &key, size_t size) {
  std::string value = this->get_metadata(key, size);
  return !value.empty() ? std::stoi(value) : 0;
}

int Llama::get_int_metadata(const std::string &model_name,
                            const std::string &key, size_t size) {
  std::string value = this->get_metadata(model_name, key, size);
  return !value.empty() ? std::stoi(value) : 0;
}

float Llama::get_float_metadata(const std::string &key, size_t size) {
  std::string value = this->get_metadata(key, size);
  return !value.empty() ? std::stof(value) : 0.0;
}

float Llama::get_float_metadata(const std::string &model_name,
                                const std::string &key, size_t size) {
  std::string value = this->get_metadata(model_name, key, size);
  return !value.empty() ? std::stof(value) : 0.0;
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
      this->get_metadata("general.architecture", 32);
  metadata.general.quantization_version =
      this->get_int_metadata("general.quantization_version", 4);
  metadata.general.alignment = this->get_int_metadata("general.alignment", 4);

  // general metadata
  metadata.general.name = this->get_metadata("general.name", 32);
  metadata.general.author = this->get_metadata("general.author", 32);
  metadata.general.version = this->get_metadata("general.version", 32);
  metadata.general.organization =
      this->get_metadata("general.organization", 32);

  metadata.general.basename = this->get_metadata("general.basename", 32);
  metadata.general.finetune = this->get_metadata("general.finetune", 32);
  metadata.general.description = this->get_metadata("general.description", 512);
  metadata.general.quantized_by = this->get_metadata("quantized_by", 32);
  metadata.general.size_label = this->get_metadata("general.size_label", 32);

  metadata.general.license = this->get_metadata("general.license", 32);
  metadata.general.license_name =
      this->get_metadata("general.license.name", 32);
  metadata.general.license_link =
      this->get_metadata("general.license.link", 32);

  metadata.general.url = this->get_metadata("general.url", 128);
  metadata.general.repo_url = this->get_metadata("general.repo_url", 128);
  metadata.general.doi = this->get_metadata("general.doi", 64);
  metadata.general.uuid = this->get_metadata("general.uuid", 64);

  std::string file_type = this->get_metadata("general.file_type", 32);
  if (gguf_types.find(file_type) != gguf_types.end()) {
    metadata.general.file_type = gguf_types.at(file_type);
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
  metadata.model.tensor_data_layout = this->get_metadata(
      metadata.general.architecture, ".tensor_data_layout", 16);

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

  metadata.model.rope.scaling_type = this->get_metadata(
      metadata.general.architecture, ".rope.scaling.type", 16);
  metadata.model.rope.scaling_factor = this->get_float_metadata(
      metadata.general.architecture, ".rope.scaling.factor", 16);
  metadata.model.rope.scaling_original_context_length =
      this->get_int_metadata(metadata.general.architecture,
                             ".rope.scaling.original_context_length", 16);
  metadata.model.rope.scaling_finetuned =
      this->get_metadata(metadata.general.architecture,
                         ".rope.scaling.finetuned", 8) == "true";

  // tokenizer metadata
  metadata.tokenizer.model = this->get_metadata("tokenizer.ggml.model", 32);

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
  metadata.tokenizer.chat_template =
      this->get_metadata("tokenizer.chat_template", 4096);

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

void Llama::cancel_goal(uint64_t goal_id) {
  auto slot = this->slot_manager_->get_slot_by_gid(goal_id);
  if (slot != nullptr) {
    slot->stop = CANCEL;
  }
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
                         std::vector<std::string> stop, bool reset) {
  auto slot = this->slot_manager_->get_slot_by_gid(slot_gid);
  if (!slot) {
    return Result<ServerTaskResultCompletion>::error(
        "Slot not found for given ID");
  }

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
  this->slot_manager_->release_slot(slot);
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

  // check the limits (n_predict)
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

  if (slot->n_predict < 1 && slot->params.n_predict < 1 &&
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
  bool has_self_spec = (spec_params.type != COMMON_SPECULATIVE_TYPE_NONE &&
                        spec_params.type != COMMON_SPECULATIVE_TYPE_DRAFT &&
                        spec_params.type != COMMON_SPECULATIVE_TYPE_EAGLE3);

  if (!has_draft && !has_self_spec) {
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

  // Check compatibility
  if (!common_speculative_is_compat(this->ctx)) {
    LLAMA_LOG_WARN("Target context is not compatible with speculative "
                   "decoding, skipping");
    return;
  }

  // Load draft model if using draft-based speculative decoding
  if (has_draft) {
    LLAMA_LOG_INFO("Loading draft model for speculative decoding: %s",
                   spec_params.mparams_dft.path.c_str());

    // Prepare params for draft model loading
    common_params params_dft;
    params_dft.model.path = spec_params.mparams_dft.path;
    params_dft.n_gpu_layers = spec_params.n_gpu_layers;
    if (spec_params.n_ctx > 0) {
      params_dft.n_ctx = spec_params.n_ctx;
    }

    // Use target model's thread settings if not overridden
    if (spec_params.cpuparams.n_threads <= 0) {
      params_dft.cpuparams.n_threads = this->params.cpuparams.n_threads;
      params_dft.cpuparams_batch.n_threads =
          this->params.cpuparams_batch.n_threads;
    } else {
      params_dft.cpuparams = spec_params.cpuparams;
      params_dft.cpuparams_batch = spec_params.cpuparams_batch;
    }

    auto mparams_dft = common_model_params_to_llama(params_dft);
    this->model_dft_ =
        llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft);

    if (this->model_dft_ == nullptr) {
      LLAMA_LOG_ERROR("Failed to load draft model '%s', speculative decoding "
                      "will be disabled",
                      params_dft.model.path.c_str());
      return;
    }

    spec_params.model_dft = this->model_dft_;
    spec_params.cparams_dft = common_context_params_to_llama(params_dft);

    // Infer speculative type from draft model if type is "none"
    if (spec_params.type == COMMON_SPECULATIVE_TYPE_NONE) {
      spec_params.type = COMMON_SPECULATIVE_TYPE_DRAFT;
    }

    LLAMA_LOG_INFO("Draft model loaded successfully");
  }

  // Initialize speculative decoder
  this->speculative_ = common_speculative_init(spec_params, this->ctx);

  if (this->speculative_ == nullptr) {
    LLAMA_LOG_ERROR("Failed to initialize speculative decoder");
    if (this->model_dft_ != nullptr) {
      llama_model_free(this->model_dft_);
      this->model_dft_ = nullptr;
    }
    return;
  }

  LLAMA_LOG_INFO("Speculative decoding initialized (type: %s, n_max: %d, "
                 "n_min: %d, p_min: %.2f)",
                 common_speculative_type_to_str(spec_params.type).c_str(),
                 spec_params.n_max, spec_params.n_min, spec_params.p_min);
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
  llama_tokens draft = common_speculative_draft(this->speculative_, spec_params,
                                                prompt_tgt, id_last);

  // Build batch: [id_last, draft0, draft1, ..., draftN-1]
  common_batch_clear(this->batch);
  common_batch_add(this->batch, id_last, slot->n_past, {slot->id}, true);

  // Skip small drafts
  if ((int)draft.size() < spec_params.n_min) {
    draft.clear();
  }

  for (size_t i = 0; i < draft.size(); ++i) {
    common_batch_add(this->batch, draft[i], slot->n_past + 1 + i, {slot->id},
                     true);
  }

  // Decode the batch on the target model
  const int ret = llama_decode(this->ctx, this->batch);
  if (ret != 0) {
    LLAMA_LOG_ERROR("Speculative decode failed with error %d", ret);
    slot->stop = ABORT;
    slot->has_next_token = false;
    return false;
  }

  // Verify draft tokens using the target sampler
  const auto ids =
      common_sampler_sample_and_accept_n(slot->sampler, this->ctx, draft);

  // ids always has at least 1 token (the one the target model would have
  // sampled) ids.size()-1 draft tokens were accepted

  const int n_accepted = (int)ids.size() - 1;
  common_speculative_accept(this->speculative_, n_accepted);

  LLAMA_LOG_DEBUG("Speculative: drafted %d, accepted %d/%d", (int)draft.size(),
                  n_accepted, (int)draft.size());

  // Process accepted tokens + the final sampled token
  slot->n_past += ids.size();
  bool should_continue = true;

  for (size_t i = 0; i < ids.size(); ++i) {
    // Update prompt_tgt for future calls
    prompt_tgt.push_back(id_last);
    id_last = ids[i];

    // Check for end of generation
    if (llama_vocab_is_eog(this->get_vocab(), id_last)) {
      slot->stop = FULL_STOP;
      slot->has_next_token = false;
      should_continue = false;

      LLAMA_LOG_INFO("Speculative: stopped by EOS");
      break;
    }

    // Build CompletionOutput for this token
    CompletionOutput result;
    result.token = id_last;
    result.text_to_send = common_token_to_piece(this->ctx, id_last);
    result.probs = this->get_probs(slot);
    slot->generated_probs.push_back(result.probs);

    slot->n_decoded += 1;

    // Run process_token to handle stop words, limits, etc.
    if (!this->process_token(slot, &result)) {
      should_continue = false;
      break;
    }
  }

  // Clear KV cache for any extra draft tokens that were rejected
  llama_memory_seq_rm(llama_get_memory(this->ctx), slot->id, slot->n_past, -1);

  if (!should_continue) {
    this->send_completion_result(slot);
    this->release_slot(slot);
  }

  return should_continue;
}

void Llama::run_loop() {
  while (!this->canceled) {

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

    // apply context shift
    if (this->params.ctx_shift) {
      for (auto &slot : this->server_slots) {
        if (slot.state != SLOT_STATE_GENERATING) {
          continue;
        }

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
    common_batch_clear(this->batch);

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

      slot.i_batch = this->batch.n_tokens;
      common_batch_add(this->batch, slot.sampled, slot.n_past, {slot.id}, true);

      slot.n_past += 1;
    }

    // Process prompts (new inputs)
    int32_t n_batch = llama_n_batch(this->ctx);
    if (this->params.cont_batching || this->batch.n_tokens == 0) {
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

          // always start from zero KV
          slot.n_past = 0;

          slot.n_prompt_tokens = prompt_tokens.size();
          slot.state = SLOT_STATE_PROCESSING_PROMPT;

          // empty prompt -> release and send empty response
          if (prompt_tokens.empty()) {
            LLAMA_LOG_WARN("Empty prompt on slot %d", slot.id);
            this->fail_pending(slot.goal_id, "Empty prompt");
            this->release_slot(&slot);
            continue;
          }

          // must fit into one ubatch and within context
          if (static_cast<uint32_t>(slot.n_prompt_tokens) >
              llama_n_ubatch(this->ctx)) {
            LLAMA_LOG_WARN("Prompt too long for slot %d, %d tokens, max %d",
                           slot.id, slot.n_prompt_tokens,
                           llama_n_ubatch(this->ctx));
            this->fail_pending(slot.goal_id, "Prompt too long");
            this->release_slot(&slot);
            continue;
          }

          if (slot.n_prompt_tokens > slot.n_ctx) {
            LLAMA_LOG_WARN("Prompt exceeds context size for slot %d", slot.id);
            this->fail_pending(slot.goal_id, "Prompt exceeds context size");
            this->release_slot(&slot);
            continue;
          }

          // wipe any previous KV for this seq
          llama_memory_seq_rm(llama_get_memory(this->ctx), slot.id, -1, -1);

          // ensure at least one token will be evaluated
          if (slot.n_past == slot.n_prompt_tokens && slot.n_past > 0) {
            slot.n_past--;
          }

          slot.n_prompt_tokens_processed = 0;
        }

        // enforce prompt fits in current batch too
        if (static_cast<uint32_t>(this->batch.n_tokens + slot.n_prompt_tokens) >
            llama_n_batch(this->ctx)) {
          continue;
        }

        // process MTMD chunks if present
        if (slot.n_past < slot.n_prompt_tokens &&
            slot.prompt_tokens[slot.n_past] == LLAMA_TOKEN_NULL) {
          process_mtmd_chunk(&slot);
        }

        // enqueue all prompt tokens (must fit fully in one batch)
        while (slot.n_past < slot.n_prompt_tokens) {
          llama_token cur_tok = slot.prompt_tokens[slot.n_past];
          if (cur_tok == LLAMA_TOKEN_NULL) {
            break; // end of text chunk
          }
          const bool need_embd = this->is_embedding() || this->is_reranking();

          common_batch_add(this->batch, cur_tok, slot.n_past, {slot.id},
                           need_embd);

          slot.n_prompt_tokens_processed++;
          slot.n_past++;
        }

        LLAMA_LOG_INFO("Added %d prompt tokens for slot %d",
                       slot.n_prompt_tokens_processed, slot.id);

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
          this->batch.logits[this->batch.n_tokens - 1] = true;

          slot.n_decoded = 0;
          slot.i_batch = this->batch.n_tokens - 1;

          LLAMA_LOG_INFO(
              "prompt done (no caching), n_past = %d, n_tokens = %d\n",
              slot.n_past, this->batch.n_tokens);
        }

        if (static_cast<uint32_t>(this->batch.n_tokens) >=
            llama_n_batch(this->ctx)) {
          LLAMA_LOG_ERROR("Batch full after adding prompts");
          continue;
        }
      }
    }

    // Check if there are no tokens to decode
    if (this->batch.n_tokens == 0) {
      LLAMA_LOG_ERROR("No tokens to decode in this iteration");
      continue;
    }

    int32_t i_next = 0;

    LLAMA_LOG_DEBUG("Decoding batch of %d tokens", this->batch.n_tokens);
    for (int32_t i = 0; i < this->batch.n_tokens; i = i_next) {
      const int32_t n_tokens = std::min(n_batch, this->batch.n_tokens - i);

      llama_batch batch_view = {
          n_tokens,
          this->batch.token + i,
          nullptr,
          this->batch.pos + i,
          this->batch.n_seq_id + i,
          this->batch.seq_id + i,
          this->batch.logits + i,
      };

      const int ret = llama_decode(this->ctx, batch_view);

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
          LLAMA_LOG_ERROR("Decoding error: %s", err.c_str());
          this->cancel();
          break; // abort the decode loop
        }

        // No readable error - likely KV pressure: backoff and retry smaller
        // batch window
        n_batch = std::max(1, n_batch / 2);
        continue; // retry current window with smaller n_batch
      }

      i_next = i + n_tokens;
      n_batch = llama_n_batch(this->ctx);

      // Consume results per-slot for the tokens we just decoded
      for (auto &slot : this->server_slots) {
        if (slot.i_batch < (int)i || slot.i_batch >= (int)(i + n_tokens)) {
          continue;
        }

        // If we just finished prompt eval for this slot, branch by task type
        if (slot.state == SLOT_STATE_DONE_PROMPT) {
          if (slot.task_type == SERVER_TASK_TYPE_EMBEDDING) {
            this->send_embedding_result(&slot, batch_view);
            this->release_slot(&slot);
            slot.i_batch = -1;
            continue;
          }

          if (slot.task_type == SERVER_TASK_TYPE_RERANK) {
            this->send_rerank_result(&slot, batch_view);
            this->release_slot(&slot);
            slot.i_batch = -1;
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
            common_speculative_begin(this->speculative_, prompt_tgt);

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
void Llama::send_embedding_result(ServerSlot *slot,
                                  const llama_batch & /*batch*/) {
  auto result = std::make_unique<ServerTaskResultEmbedding>();
  result->id_slot = slot->id;
  result->id = slot->goal_id;
  result->n_tokens = this->batch.n_tokens;
  const int n_embd = llama_model_n_embd(this->model);

  std::vector<float> embd_res(n_embd, 0.0f);

  for (int i = 0; i < this->batch.n_tokens; ++i) {
    if (!this->batch.logits[i] || this->batch.seq_id[i][0] != slot->id) {
      continue;
    }

    const float *embd = nullptr;
    if (llama_pooling_type(this->ctx) == LLAMA_POOLING_TYPE_NONE) {
      embd = llama_get_embeddings_ith(this->ctx, i);
    } else {
      embd = llama_get_embeddings_seq(this->ctx, this->batch.seq_id[i][0]);
    }

    if (embd == nullptr) {
      LLAMA_LOG_ERROR("failed to get embeddings, token = %d, seq_id = %d\n",
                      this->batch.token[i], this->batch.seq_id[i][0]);

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

void Llama::send_rerank_result(ServerSlot *slot,
                               const llama_batch & /*batch*/) {
  auto result = std::make_unique<ServerTaskResultRerank>();
  result->id_slot = slot->id;
  result->id = slot->goal_id;
  for (int i = 0; i < this->batch.n_tokens; ++i) {
    if (!this->batch.logits[i] || this->batch.seq_id[i][0] != slot->id) {
      continue;
    }

    const float *embd =
        llama_get_embeddings_seq(this->ctx, this->batch.seq_id[i][0]);
    if (embd == NULL) {
      embd = llama_get_embeddings_ith(this->ctx, i);
    }

    if (embd == NULL) {
      LLAMA_LOG_ERROR("failed to get embeddings, token = %d, seq_id = %d\n",
                      this->batch.token[i], this->batch.seq_id[i][0]);

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
      "b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT;
  task_result->oaicompat_model = this->get_metadata().general.name;
  task_result->oaicompat_cmpl_id = llama_utils::gen_chatcmplid();
  task_result->n_decoded = slot->n_decoded;
  task_result->n_prompt_tokens = slot->n_prompt_tokens;
  task_result->oaicompat_msg =
      slot->update_chat_msg(task_result->oaicompat_msg_diffs);

  const auto id = task_result->id;
  this->fulfill_pending(id, std::move(task_result));
}
