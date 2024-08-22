// MIT License

// Copyright (c) 2023  Miguel Ángel González Santamarta

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

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

#include "common.h"
#include "grammar-parser.h"
#include "llama.h"
#include "llama_utils/spinner.hpp"

// llama logs
#define LLAMA_LOG_ERROR(text, ...)                                             \
  fprintf(stderr, "[ERROR] " text "\n", ##__VA_ARGS__)
#define LLAMA_LOG_WARN(text, ...)                                              \
  fprintf(stderr, "[WARN] " text "\n", ##__VA_ARGS__)
#define LLAMA_LOG_INFO(text, ...)                                              \
  fprintf(stderr, "[INFO] " text "\n", ##__VA_ARGS__)

// llama structs
struct token_prob {
  llama_token token;
  float probability;
};

struct completion_output {
  std::vector<token_prob> probs;
  llama_token token;
};

enum stop_type {
  NO_STOP,
  FULL_STOP,
  PARTIAL_STOP,
  CANCEL,
  ABORT,
};

struct response_output {
  std::vector<completion_output> completions;
  stop_type stop;
};

struct embeddings_ouput {
  std::vector<float> embeddings;
  int32_t n_tokens;
};

namespace llama_ros {

using GenerateResponseCallback = std::function<void(struct completion_output)>;

class Llama {

public:
  Llama(const struct gpt_params &params, bool debug);
  virtual ~Llama();

  std::vector<llama_token> tokenize(const std::string &text, bool add_bos,
                                    bool special = false);
  std::string detokenize(const std::vector<llama_token> &tokens);

  void reset();
  void cancel();

  embeddings_ouput generate_embeddings(const std::string &input_prompt,
                                       bool normalize = true);
  response_output generate_response(const std::string &input_prompt,
                                    struct llama_sampling_params sparams,
                                    bool ignore_eos = false,
                                    GenerateResponseCallback callbakc = nullptr,
                                    std::vector<std::string> stop = {});
  response_output generate_response(const std::string &input_prompt,
                                    bool ignore_eos = false,
                                    GenerateResponseCallback callbakc = nullptr,
                                    std::vector<std::string> stop = {});

  const struct llama_context *get_ctx() { return this->ctx; }
  int get_n_ctx() { return llama_n_ctx(this->ctx); }
  int get_n_ctx_train() { return llama_n_ctx_train(this->model); }
  int get_n_embd() { return llama_n_embd(this->model); }
  int get_n_vocab() { return llama_n_vocab(this->model); }
  bool is_embedding() { return this->params.embedding; }
  bool add_bos_token() { return llama_add_bos_token(this->model); }
  llama_token get_token_eos() { return llama_token_eos(this->model); }

protected:
  struct gpt_params params;

  // model
  struct llama_context *ctx;
  struct llama_model *model;
  struct llama_sampling_context *ctx_sampling;

  // aux
  bool debug;
  bool canceled;
  llama_utils::Spinner spinner;
  std::vector<llama_token> prompt_tokens;

  // eval
  int32_t n_past;
  int32_t n_remain;
  int32_t n_consumed;
  int32_t ga_i;

  virtual void load_prompt(const std::string &input_prompt, bool add_pfx,
                           bool add_sfx);

  stop_type
  find_stop(std::vector<struct completion_output> completion_result_list,
            std::vector<std::string> stopping_words);
  stop_type
  find_stop_word(std::vector<struct completion_output> completion_result_list,
                 std::string stopping_word);

  bool eval_system_prompt();
  virtual bool eval_prompt();
  bool eval_prompt(std::vector<llama_token> prompt_tokens);
  bool eval_token(llama_token token);
  bool eval(std::vector<llama_token> tokens);
  bool eval(struct llama_batch batch);

  std::vector<token_prob> get_probs();
  struct completion_output sample();
  void update_sampling_context(const struct llama_sampling_params &params);

private:
  // lock
  std::recursive_mutex mutex;
};

} // namespace llama_ros

#endif
