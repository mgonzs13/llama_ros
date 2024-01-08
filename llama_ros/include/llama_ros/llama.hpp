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
#include "common/grammar-parser.h"
#include "llama.h"
#include "llama_ros/spinner.hpp"

struct completion_output {
  struct token_prob {
    llama_token token;
    float probability;
  };

  std::vector<token_prob> probs;
  llama_token token;
};

namespace llama_ros {

class Llama {

  using GenerateResponseCallback =
      std::function<void(struct completion_output)>;

public:
  Llama(const struct gpt_params &params, bool debug);
  ~Llama();

  std::vector<llama_token> tokenize(const std::string &text, bool add_bos,
                                    bool special = false);
  std::string detokenize(const std::vector<llama_token> &tokens);
  void reset();
  void cancel();
  std::vector<float> generate_embeddings(const std::string &input_prompt);
  std::vector<struct completion_output>
  generate_response(const std::string &input_prompt, bool add_pfx_sfx = true,
                    GenerateResponseCallback callbakc = nullptr);

  const struct llama_context *get_ctx() { return this->ctx; }
  struct gpt_params &get_params() {
    return this->params;
  }
  int get_n_ctx() { return llama_n_ctx(this->ctx); }
  int get_n_ctx_train() { return llama_n_ctx_train(this->model); }
  int get_n_embd() { return llama_n_embd(this->model); }
  int get_n_vocab() { return llama_n_vocab(this->model); }
  bool is_embedding() { return this->params.embedding; }
  bool should_add_bos_token() {
    return llama_should_add_bos_token(this->model);
  }
  llama_token get_token_eos() { return llama_token_eos(this->model); }

protected:
  struct llama_model *model;
  struct llama_context *ctx;
  struct llama_sampling_context *ctx_sampling;

  bool eval();
  struct completion_output sample();
  void update_sampling_params(const struct llama_sampling_params &params);

private:
  bool debug;
  Spinner spinner;
  struct gpt_params params;

  // aux
  std::vector<llama_token> prompt_tokens;
  std::vector<llama_token> batch_tokens;

  // eval
  bool canceled;
  int32_t n_past;
  int32_t n_remain;
  int32_t n_consumed;
  int32_t ga_i;

  // lock
  std::recursive_mutex mutex;
};

} // namespace llama_ros

#endif
