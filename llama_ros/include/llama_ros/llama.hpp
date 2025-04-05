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

#ifndef LLAMA_ROS__LLAMA_HPP
#define LLAMA_ROS__LLAMA_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"

#include "llama_utils/spinner.hpp"

namespace llama_ros {

// llama structs
struct TokenProb {
  llama_token token;
  float probability;
};

struct LoRA {
  int id;
  std::string path;
  float scale;
};

struct CompletionOutput {
  std::vector<TokenProb> probs;
  llama_token token;
};

enum StopType {
  NO_STOP,
  FULL_STOP,
  PARTIAL_STOP,
  CANCEL,
  ABORT,
};

struct ResponseOutput {
  std::vector<CompletionOutput> completions;
  StopType stop;
};

struct EmbeddingsOuput {
  std::vector<float> embeddings;
  int32_t n_tokens;
};

struct Metadata {
  struct GeneralInfo {
    std::string architecture;
    uint32_t quantization_version;
    uint32_t alignment;

    std::string name;
    std::string author;
    std::string version;
    std::string organization;

    std::string basename;
    std::string finetune;
    std::string description;
    std::string quantized_by;
    std::string size_label;

    std::string license;
    std::string license_name;
    std::string license_link;

    std::string url;
    std::string repo_url;
    std::string doi;
    std::string uuid;

    std::vector<std::string> tags;
    std::vector<std::string> languages;
    std::vector<std::string> datasets;
    std::string file_type;
  };

  struct AttentionInfo {
    uint64_t head_count;
    uint64_t head_count_kv;

    float max_alibi_bias;
    float clamp_kqv;

    float layer_norm_epsilon;
    float layer_norm_rms_epsilon;

    uint32_t key_length;
    uint32_t value_length;
  };

  struct RoPEInfo {
    uint64_t dimension_count;
    float freq_base;

    std::string scaling_type;
    float scaling_factor;
    uint32_t scaling_original_context_length;
    bool scaling_finetuned;
  };

  struct ModelInfo {
    uint64_t context_length;
    uint64_t embedding_length;
    uint64_t block_count;
    uint64_t feed_forward_length;

    bool use_parallel_residual;
    std::string tensor_data_layout;

    uint32_t expert_count;
    uint32_t expert_used_count;

    AttentionInfo attention;
    RoPEInfo rope;
  };

  struct TokenizerInfo {
    std::string model;

    uint32_t bos_token_id;
    uint32_t eos_token_id;
    uint32_t unknown_token_id;
    uint32_t padding_token_id;
    uint32_t separator_token_id;
    bool add_bos_token;

    std::string chat_template;
  };

  GeneralInfo general;
  ModelInfo model;
  TokenizerInfo tokenizer;
};

using GenerateResponseCallback = std::function<void(struct CompletionOutput)>;

class Llama {

public:
  Llama(const struct common_params &params, std::string system_prompt = "",
        bool initial_reset = true);
  virtual ~Llama();

  std::vector<llama_token> tokenize(const std::string &text, bool add_bos,
                                    bool special = false);
  std::string detokenize(const std::vector<llama_token> &tokens);

  virtual void reset();
  void cancel();

  std::vector<struct LoRA> list_loras();
  void update_loras(std::vector<struct LoRA> loras);

  std::vector<llama_token>
  truncate_tokens(const std::vector<llama_token> &tokens, int limit_size,
                  bool add_eos = true);
  struct EmbeddingsOuput generate_embeddings(const std::string &input_prompt,
                                             int normalization = 2);
  struct EmbeddingsOuput
  generate_embeddings(const std::vector<llama_token> &tokens,
                      int normalization = 2);
  float rank_document(const std::string &query, const std::string &document);
  std::vector<float> rank_documents(const std::string &query,
                                    const std::vector<std::string> &documents);

  struct ResponseOutput
  generate_response(const std::string &input_prompt,
                    struct common_params_sampling sparams,
                    GenerateResponseCallback callbakc = nullptr,
                    std::vector<std::string> stop = {});
  struct ResponseOutput
  generate_response(const std::string &input_prompt,
                    GenerateResponseCallback callbakc = nullptr,
                    std::vector<std::string> stop = {});

  struct std::unique_ptr<struct common_chat_templates,
                         common_chat_templates_deleter>
  get_chat_templates();
  struct common_chat_params
  get_chat_params(struct common_chat_templates *tmpls,
                  struct common_chat_templates_inputs inputs);
  struct llama_perf_context_data get_perf_data();

  const struct llama_context *get_ctx() { return this->ctx; }
  const struct llama_model *get_model() { return this->model; }
  const struct llama_vocab *get_vocab() {
    return llama_model_get_vocab(this->model);
  }

  int get_n_ctx() { return llama_n_ctx(this->ctx); }
  int get_n_ctx_train() { return llama_model_n_ctx_train(this->model); }
  int get_n_embd() { return llama_model_n_embd(this->model); }
  int get_n_vocab() { return llama_vocab_n_tokens(this->get_vocab()); }

  std::string get_metadata(const std::string &key, size_t size);
  std::string get_metadata(const std::string &model_name,
                           const std::string &key, size_t size);
  int get_int_metadata(const std::string &key, size_t size);
  int get_int_metadata(const std::string &model_name, const std::string &key,
                       size_t size);
  float get_float_metadata(const std::string &key, size_t size);
  float get_float_metadata(const std::string &model_name,
                           const std::string &key, size_t size);
  struct Metadata get_metadata();

  bool is_embedding() { return this->params.embedding; }
  bool is_reranking() { return this->params.reranking; }

  bool add_bos_token() { return llama_vocab_get_add_bos(this->get_vocab()); }
  bool is_eog() {
    return llama_vocab_is_eog(this->get_vocab(),
                              common_sampler_last(this->sampler));
  }
  llama_token get_token_eos() { return llama_vocab_eos(this->get_vocab()); }
  llama_token get_token_bos() { return llama_vocab_bos(this->get_vocab()); }
  llama_token get_token_sep() { return llama_vocab_sep(this->get_vocab()); }

protected:
  struct common_params params;

  // model
  struct common_init_result llama_init;
  struct llama_context *ctx;
  struct llama_model *model;
  std::vector<common_adapter_lora_info> lora_adapters;
  struct common_sampler *sampler;
  struct ggml_threadpool *threadpool;
  struct ggml_threadpool *threadpool_batch;

  // aux
  std::string system_prompt;
  bool canceled;
  llama_utils::Spinner spinner;
  std::vector<llama_token> prompt_tokens;

  // eval
  int32_t n_past;
  int32_t n_consumed;
  int32_t ga_i;

  virtual void load_prompt(const std::string &input_prompt, bool add_pfx,
                           bool add_sfx);

  StopType
  find_stop(std::vector<struct CompletionOutput> completion_result_list,
            std::vector<std::string> stopping_words);
  StopType
  find_stop_word(std::vector<struct CompletionOutput> completion_result_list,
                 std::string stopping_word);

  bool eval_system_prompt();
  virtual bool eval_prompt();
  bool eval_prompt(std::vector<llama_token> prompt_tokens);
  bool eval_token(llama_token token);
  bool eval(std::vector<llama_token> tokens);
  virtual bool eval(struct llama_batch batch);

  std::vector<struct TokenProb> get_probs();
  struct CompletionOutput sample();

private:
  // lock
  std::recursive_mutex mutex;
};

} // namespace llama_ros

#endif
