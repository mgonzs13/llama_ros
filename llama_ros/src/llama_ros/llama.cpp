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
#include <cmath>
#include <map>
#include <memory>

#include "common.h"
#include "sampling.h"

#include "llama_ros/llama.hpp"
#include "llama_utils/logs.hpp"

using namespace llama_ros;

Llama::Llama(const struct common_params &params, std::string system_prompt,
             bool initial_reset)
    : params(params), system_prompt(system_prompt) {

  print_build_info();

  // load model
  llama_backend_init();
  llama_numa_init(this->params.numa);

  this->llama_init = common_init_from_params(this->params);
  this->model = llama_init.model.get();
  this->ctx = llama_init.context.get();
  this->lora_adapters = this->params.lora_adapters;

  if (this->model == NULL) {
    LLAMA_LOG_ERROR("Unable to load model");
    return;
  }

  // init threadpool
  LLAMA_LOG_INFO("llama threadpool init = n_threads = %d",
                 this->params.cpuparams.n_threads);

  struct ggml_threadpool_params tpp_batch =
      ggml_threadpool_params_from_cpu_params(this->params.cpuparams_batch);
  struct ggml_threadpool_params tpp =
      ggml_threadpool_params_from_cpu_params(this->params.cpuparams);

  set_process_priority(this->params.cpuparams.priority);

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

  this->threadpool = ggml_threadpool_new(&tpp);
  if (!this->threadpool) {
    LLAMA_LOG_ERROR("Failed to create threadpool: n_threads %d", tpp.n_threads);
    return;
  }

  llama_attach_threadpool(this->ctx, this->threadpool, this->threadpool_batch);

  // create the sampler
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

  // set inital values
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
}

Llama::~Llama() {
  llama_free(this->ctx);
  this->ctx = nullptr;

  llama_model_free(this->model);
  this->model = nullptr;

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

  llama_kv_self_clear(this->ctx);

  if (this->sampler != nullptr) {
    common_sampler_reset(this->sampler);
  }

  this->canceled = false;
  this->n_past = 0;
  this->n_consumed = 0;
  this->ga_i = 0;

  this->prompt_tokens.clear();

  // load system prompt
  if (!this->eval_system_prompt()) {
    LLAMA_LOG_ERROR("Failed to eval system prompt");
  }

  // number of tokens to keep when resetting context
  if (this->params.n_keep < 0) {
    this->params.n_keep = (int)this->prompt_tokens.size();
  }
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

struct Metadata Llama::get_metadata() {

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

  struct Metadata metadata;

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
  if (gguf_types.find(file_type) == gguf_types.end()) {
    metadata.general.file_type = gguf_types.at(file_type.c_str());
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
  std::lock_guard<std::recursive_mutex> lk(this->mutex);
  return common_tokenize(this->ctx, text, add_bos, special);
}

std::string Llama::detokenize(const std::vector<llama_token> &tokens) {
  std::lock_guard<std::recursive_mutex> lk(this->mutex);

  std::string text;

  for (llama_token t : tokens) {
    text.append(common_token_to_piece(this->ctx, t));
  }

  return text;
}

void Llama::cancel() { this->canceled = true; }

/*
*******************************
*         EMBEDDINGS          *
*******************************
*/
struct EmbeddingsOuput
Llama::generate_embeddings(const std::vector<llama_token> &tokens,
                           int normalization) {
  std::lock_guard<std::recursive_mutex> lk(this->mutex);

  const int n_embd = this->get_n_embd();

  struct EmbeddingsOuput output;
  output.embeddings = std::vector<float>(n_embd, 0.0f);
  output.n_tokens = 0;

  if (!this->is_embedding()) {
    LLAMA_LOG_ERROR(
        "Llama must be created with embedding enable to create embeddings");
    return output;
  }

  if ((int)tokens.size() > this->get_n_ctx()) {
    LLAMA_LOG_ERROR("Prompt too long %ld, context size is %d", tokens.size(),
                    this->get_n_ctx());
    return output;
  }

  // llama eval
  struct llama_batch batch = llama_batch_init(this->params.n_batch, 0, 1);
  for (size_t i = 0; i < tokens.size(); i++) {
    common_batch_add(batch, tokens[i], i, {0}, i == tokens.size() - 1);
  }

  if (llama_encode(this->ctx, batch)) {
    LLAMA_LOG_ERROR("Failed to eval");
    return output;
  }

  // get embeddings
  std::vector<float> embd_res(n_embd, 0.0f);

  for (int i = 0; i < batch.n_tokens; ++i) {

    if (!batch.logits[i]) {
      continue;
    }

    const float *embd = llama_get_embeddings_seq(this->ctx, batch.seq_id[i][0]);
    if (embd == NULL) {
      embd = llama_get_embeddings_ith(this->ctx, i);
    }

    if (embd == NULL) {
      LLAMA_LOG_ERROR("Failed to get embeddings");
      continue;
    }

    common_embd_normalize(embd, embd_res.data(), n_embd, normalization);
  }

  // clear
  llama_kv_self_seq_rm(this->ctx, 0, 0, -1);
  llama_batch_free(batch);

  // result
  output.embeddings = embd_res;
  output.n_tokens = tokens.size();

  return output;
}

struct EmbeddingsOuput
Llama::generate_embeddings(const std::string &input_prompt, int normalization) {

  auto tokens = this->tokenize(input_prompt, this->add_bos_token(), true);
  tokens = this->truncate_tokens(tokens, this->params.n_batch, true);

  return this->generate_embeddings(tokens, normalization);
}

std::vector<llama_token>
Llama::truncate_tokens(const std::vector<llama_token> &tokens, int limit_size,
                       bool add_eos) {

  std::vector<llama_token> new_tokens = tokens;

  if ((int)tokens.size() > limit_size) {
    LLAMA_LOG_WARN("Prompt too long %ld, limit size %d, truncating...",
                   tokens.size(), limit_size);
    new_tokens.resize(limit_size);
  }

  // add eos if not present
  if (add_eos && tokens.back() != this->get_token_eos()) {
    new_tokens.push_back(this->get_token_eos());
  }

  return new_tokens;
}

/*
*****************************
*         RERANKING         *
*****************************
*/
float Llama::rank_document(const std::string &query,
                           const std::string &document) {

  if (!this->is_reranking()) {
    LLAMA_LOG_ERROR(
        "Llama must be created with reranking enable to make rerank");
    return 0.0;
  }

  std::vector<llama_token> tokens;
  tokens.reserve(this->params.n_batch);

  tokens.push_back(this->get_token_bos());

  auto part1 = this->tokenize(query, false, true);
  part1 =
      this->truncate_tokens(part1, (int)(this->params.n_batch / 2) - 2, true);
  tokens.insert(tokens.end(), part1.begin(), part1.end());

  tokens.push_back(this->get_token_eos());
  tokens.push_back(this->get_token_sep());

  auto part2 = this->tokenize(document, false, true);
  part2 =
      this->truncate_tokens(part2, (int)(this->params.n_batch / 2) - 2, true);
  tokens.insert(tokens.end(), part2.begin(), part2.end());

  tokens.push_back(this->get_token_eos());

  return this->generate_embeddings(tokens, -1).embeddings.at(0);
}

std::vector<float>
Llama::rank_documents(const std::string &query,
                      const std::vector<std::string> &documents) {

  if (!this->is_reranking()) {
    LLAMA_LOG_ERROR(
        "Llama must be created with reranking enable to make rerank");
    return {0.0};
  }

  std::vector<float> scores;

  for (std::string doc : documents) {
    scores.push_back(this->rank_document(query, doc));
  }

  return scores;
}

/*
*******************************
*            LORAS            *
*******************************
*/
std::vector<struct LoRA> Llama::list_loras() {

  std::lock_guard<std::recursive_mutex> lk(this->mutex);

  std::vector<struct LoRA> loras;

  for (size_t i = 0; i < this->lora_adapters.size(); ++i) {
    auto &lora_i = this->lora_adapters[i];

    struct LoRA lora_aux;
    lora_aux.id = i;
    lora_aux.path = lora_i.path;
    lora_aux.scale = lora_i.scale;

    loras.push_back(lora_aux);
  }

  return loras;
}

void Llama::update_loras(std::vector<struct LoRA> loras) {

  std::lock_guard<std::recursive_mutex> lk(this->mutex);

  for (auto lora : loras) {
    if (lora.id >= 0 && lora.id <= (int)this->lora_adapters.size()) {

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
struct ResponseOutput
Llama::generate_response(const std::string &input_prompt,
                         GenerateResponseCallback callback,
                         std::vector<std::string> stop) {
  struct common_params_sampling sparams;
  return this->generate_response(input_prompt, sparams, callback, stop);
}

struct ResponseOutput Llama::generate_response(
    const std::string &input_prompt, struct common_params_sampling sparams,
    GenerateResponseCallback callback, std::vector<std::string> stop) {

  std::lock_guard<std::recursive_mutex> lk(this->mutex);

  this->canceled = false;
  struct ResponseOutput output;
  struct CompletionOutput completion_result;
  std::vector<struct CompletionOutput> response;
  std::vector<struct CompletionOutput> completion_result_list;

  std::vector<std::string> stop_concat;
  stop_concat.reserve(this->params.antiprompt.size() + stop.size());
  stop_concat.insert(stop_concat.end(), this->params.antiprompt.begin(),
                     this->params.antiprompt.end());
  stop_concat.insert(stop_concat.end(), stop.begin(), stop.end());

  // create sampler
  this->params.sampling = sparams;

  if (this->sampler != nullptr) {
    common_sampler_free(this->sampler);
  }

  this->sampler = common_sampler_init(this->model, this->params.sampling);

  if (this->sampler == nullptr) {
    output.stop = StopType::ABORT;
    return output;
  }

  // load prompt
  this->load_prompt(input_prompt, true, true);

  // show sampling info
  LLAMA_LOG_INFO("Sampler params: %s", this->params.sampling.print().c_str());
  LLAMA_LOG_INFO("Sampler constr: %s",
                 common_sampler_print(this->sampler).c_str());
  LLAMA_LOG_INFO("Prompt tokens:\n%s",
                 this->detokenize(this->prompt_tokens).c_str());
  LLAMA_LOG_INFO("Starting Response Generation");

  // eval prompt
  if (!this->eval_prompt()) {
    output.stop = StopType::ABORT;
    return output;
  }

  // generation loop
  StopType stopping = NO_STOP;

  while (stopping != FULL_STOP) {

    stopping = this->find_stop(completion_result_list, stop_concat);

    if (stopping == FULL_STOP) {
      if (this->canceled) {
        output.stop = StopType::CANCEL;
      } else {
        output.stop = StopType::FULL_STOP;
      }

      break;

    } else if (stopping == PARTIAL_STOP) {
      LLAMA_LOG_INFO("Partial stopping word found");

    } else if (stopping == NO_STOP) {
      if (!completion_result_list.empty()) {
        for (auto completion_ele : completion_result_list) {
          if (callback != nullptr) {
            callback(completion_ele);
          }
          response.push_back(completion_ele);
        }
        completion_result_list.clear();
      }
    }

    // sample next token
    completion_result = this->sample();
    completion_result_list.push_back(completion_result);

    // next eval
    if (!this->eval_token(completion_result.token)) {
      output.stop = StopType::ABORT;
      break;
    }
  }

  LLAMA_LOG_INFO("Finish Response Generation");

  common_perf_print(this->ctx, this->sampler);

  output.completions = response;
  return output;
}

/*
*****************************
*        LOAD PROMPT        *
*****************************
*/
void Llama::load_prefix() {
  std::vector<llama_token> inp_pfx = this->tokenize(
      this->params.input_prefix,
      this->add_bos_token() && this->prompt_tokens.empty(), true);

  if (!this->params.input_prefix.empty()) {

    const int n_prev = 64;
    const std::string last_output =
        common_sampler_prev_str(this->sampler, this->ctx, n_prev);

    // check if prefix is already added
    if (last_output.find(
            this->params.input_prefix.c_str(),
            last_output.length() - this->params.input_prefix.length(),
            this->params.input_prefix.length()) == std::string::npos) {

      this->prompt_tokens.insert(this->prompt_tokens.end(), inp_pfx.begin(),
                                 inp_pfx.end());
    }
  }
}

void Llama::load_suffix() {
  std::vector<llama_token> inp_sfx =
      this->tokenize(this->params.input_suffix, false, true);

  if (!this->params.input_suffix.empty()) {
    this->prompt_tokens.insert(this->prompt_tokens.end(), inp_sfx.begin(),
                               inp_sfx.end());
  }
}

void Llama::load_prompt(const std::string &input_prompt, bool add_pfx,
                        bool add_sfx) {

  std::string prompt(input_prompt);
  std::vector<llama_token> line_inp;

  if (this->prompt_tokens.empty() && !add_pfx) {
    line_inp = this->tokenize(prompt, this->add_bos_token(), true);
  } else {
    line_inp = this->tokenize(prompt, false, false);
  }

  // insert prefix
  if (add_pfx) {
    this->load_prefix();
  }

  this->prompt_tokens.insert(this->prompt_tokens.end(), line_inp.begin(),
                             line_inp.end());

  // insert suffix
  if (add_sfx) {
    this->load_suffix();
  }
}

/*
*****************************
*           STOP            *
*****************************
*/
StopType
Llama::find_stop(std::vector<struct CompletionOutput> completion_result_list,
                 std::vector<std::string> stopping_words) {

  // check if stopping word appear at the end of the output
  const int n_prev = 32;
  const std::string last_output =
      common_sampler_prev_str(this->sampler, this->ctx, n_prev);

  for (auto w : stopping_words) {
    if (last_output.find(w.c_str(), last_output.length() - w.length(),
                         w.length()) != std::string::npos) {
      LLAMA_LOG_INFO("Stopping word %s found at the end of text", w.c_str());
      return FULL_STOP;
    }
  }

  // eos
  if (this->is_eog()) {
    LLAMA_LOG_INFO("Stopping with EOS");
    return FULL_STOP;
  }

  // action server is canceled
  if (this->canceled) {
    LLAMA_LOG_INFO("Canceling llama_ros");
    return FULL_STOP;
  }

  // respect the maximum number of tokens
  if (this->n_past >= this->params.n_predict && this->params.n_predict >= 0) {
    LLAMA_LOG_INFO("Maximum number of tokens reached %d",
                   this->params.n_predict);
    return FULL_STOP;
  }

  if (this->n_past >= this->get_n_ctx() && this->params.n_predict == -2) {
    LLAMA_LOG_INFO("Maximum number of tokens reached %d", this->get_n_ctx());
    return FULL_STOP;
  }

  // search for stopping words
  for (auto w : stopping_words) {
    StopType s = this->find_stop_word(completion_result_list, w);
    if (s != NO_STOP) {

      if (s == FULL_STOP) {
        LLAMA_LOG_INFO("Stopping word %s found at the end of text", w.c_str());
      }

      return s;
    }
  }

  return NO_STOP;
}

inline std::string trim(const std::string &str) {

  // find the position of the first non-whitespace character
  size_t start = str.find_first_not_of(" \t\n\r\f\v");

  // if the string is all whitespace, return an empty string
  if (start == std::string::npos) {
    return "";
  }

  // find the position of the last non-whitespace character
  size_t end = str.find_last_not_of(" \t\n\r\f\v");

  // return the substring that excludes leading and trailing whitespace
  return str.substr(start, end - start + 1);
}

StopType Llama::find_stop_word(
    std::vector<struct CompletionOutput> completion_result_list,
    std::string stopping_word) {

  std::string completion_text = "";
  for (auto c : completion_result_list) {
    completion_text.append(trim(this->detokenize({c.token})));
  }

  if (completion_text.empty()) {
    return NO_STOP;
  }

  for (size_t i = 0; i < completion_text.size() && i < stopping_word.size();
       i++) {
    if (completion_text.at(i) != stopping_word.at(i)) {
      return NO_STOP;
    }
  }

  if (completion_text.size() >= stopping_word.size()) {
    return FULL_STOP;
  } else {
    return PARTIAL_STOP;
  }

  return NO_STOP;
}

/*
*****************************
*           EVAL            *
*****************************
*/
bool Llama::eval_system_prompt() {

  if (this->system_prompt.size() > 0) {
    // load prompt
    this->load_prompt(this->system_prompt, false, false);

    // eval prompt
    if (!this->eval_prompt()) {
      return false;
    }
  }

  return true;
}

bool Llama::eval_prompt() { return this->eval_prompt(this->prompt_tokens); }

bool Llama::eval_prompt(std::vector<llama_token> prompt_tokens) {

  std::vector<llama_token> batch;
  batch.reserve(this->params.n_batch);

  while (((int)prompt_tokens.size() > this->n_consumed)) {

    while (((int)prompt_tokens.size() > this->n_consumed) &&
           ((int)batch.size() < this->params.n_batch)) {

      batch.push_back(prompt_tokens[this->n_consumed]);
      common_sampler_accept(this->sampler, prompt_tokens[this->n_consumed],
                            false);
      ++this->n_consumed;
    }

    if (!this->eval(batch)) {
      return false;
    }

    batch.clear();
  }

  return true;
}

bool Llama::eval_token(llama_token token) {
  return this->eval(std::vector<llama_token>({token}));
}

bool Llama::eval(std::vector<llama_token> tokens) {

  // create batch
  struct llama_batch batch = {
      int32_t(tokens.size()), // n_tokens
      tokens.data(),          // tokens
      nullptr,                // embd
      nullptr,                // pos
      nullptr,                // n_seq_id
      0,                      // seq_id
      nullptr,                // logits
  };

  return this->eval(batch);
}

bool Llama::eval(struct llama_batch batch) {

  if (batch.n_tokens > 0) {

    // shift context
    if (this->params.grp_attn_n == 1) {
      if (this->n_past + batch.n_tokens > this->get_n_ctx()) {

        const int n_left = this->n_past - this->params.n_keep;
        const int n_discard = n_left / 2;

        llama_kv_self_seq_rm(this->ctx, 0, this->params.n_keep,
                             this->params.n_keep + n_discard);
        llama_kv_self_seq_add(this->ctx, 0, this->params.n_keep + n_discard,
                              n_past, -n_discard);

        this->n_past -= n_discard;
      }

    } else {
      // context extension via Self-Extend
      int ga_n = this->params.grp_attn_n;
      int ga_w = this->params.grp_attn_w;

      while (this->n_past >= this->ga_i + ga_w) {
        const int ib = (ga_n * this->ga_i) / ga_w;
        const int bd = (ga_w / ga_n) * (ga_n - 1);
        const int dd = (ga_w / ga_n) - ib * bd - ga_w;

        llama_kv_self_seq_add(this->ctx, 0, this->ga_i, this->n_past, ib * bd);
        llama_kv_self_seq_div(this->ctx, 0, this->ga_i + ib * bd,
                              this->ga_i + ib * bd + ga_w, ga_n);
        llama_kv_self_seq_add(this->ctx, 0, this->ga_i + ib * bd + ga_w,
                              this->n_past + ib * bd, dd);

        this->n_past -= bd;

        this->ga_i += ga_w / ga_n;
      }
    }

    // evaluate tokens in batches
    for (int i = 0; i < batch.n_tokens; i += this->params.n_batch) {

      int n_eval = std::min(this->params.n_batch, batch.n_tokens - i);

      struct llama_batch batch_view = {
          n_eval,
          batch.embd == nullptr ? batch.token + i : nullptr,
          batch.embd != nullptr ? batch.embd + i : nullptr,
          batch.pos + i,
          batch.n_seq_id + i,
          batch.seq_id + i,
          batch.logits + i,
      };

      this->spinner.spin("EVALUATING " + std::to_string(n_eval) + " TOKENS");

      if (llama_decode(this->ctx, batch_view)) {
        LLAMA_LOG_ERROR("Failed to eval");
        return false;
      }

      this->n_past += n_eval;
    }
  }

  return true;
}

/*
*****************************
*          SAMPLE           *
*****************************
*/
std::vector<struct TokenProb> Llama::get_probs() {
  std::vector<struct TokenProb> probs;

  const auto *cur_p = common_sampler_get_candidates(this->sampler);

  const int32_t n_probs = this->params.sampling.n_probs;

  for (int i = 0; i < n_probs; ++i) {
    probs.push_back({
        cur_p->data[i].id,
        (size_t)i >= cur_p->size ? 0.0f : cur_p->data[i].p,
    });
  }

  return probs;
}

struct CompletionOutput Llama::sample() {

  // sample token
  llama_token id = common_sampler_sample(this->sampler, this->ctx, -1);
  common_sampler_accept(this->sampler, id, true);

  // create output
  struct CompletionOutput result;
  result.token = id;
  result.probs = this->get_probs();

  // return result
  return result;
}

/*
*****************************
*   CHAT COMPLETION FUNCS   *
*****************************
*/
struct std::unique_ptr<struct common_chat_templates,
                       common_chat_templates_deleter>
Llama::get_chat_templates() {
  return std::unique_ptr<struct common_chat_templates,
                         common_chat_templates_deleter>(
      common_chat_templates_init(this->get_model(), ""));
}

struct llama_perf_context_data
Llama::get_perf_data() {
  return llama_perf_context(this->ctx);
}

struct common_chat_params
Llama::get_chat_params(struct common_chat_templates *tmpls,
                       common_chat_templates_inputs inputs) {
  return common_chat_templates_apply(tmpls, inputs);
}
