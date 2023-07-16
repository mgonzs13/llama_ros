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
#include <string>
#include <vector>

#include "llama.h"
#include "llama_ros/spinner.hpp"

struct llama_sampling_params {
  float temp;
  int32_t top_k;
  float top_p;
  float tfs_z;
  float typical_p;
  float repeat_penalty;
  int32_t repeat_last_n;
  float presence_penalty;
  float frequency_penalty;
  int32_t mirostat;
  float mirostat_tau;
  float mirostat_eta;
  bool penalize_nl;
  int32_t n_probs;
};
struct llama_sampling_params llama_sampling_default_params();

struct llama_eval_params {
  int32_t n_threads;
  int32_t n_predict;
  int32_t n_batch;
  int32_t n_keep;
};
struct llama_eval_params llama_eval_default_params();

namespace llama_ros {

class Llama {

  using GenerateResponseCallback = std::function<void(std::string)>;

public:
  Llama(llama_context_params context_params,
        const llama_eval_params &eval_params, const std::string &model,
        const std::string &lora_adapter, const std::string &lora_base,
        const bool &numa, const std::string &prefix, const std::string &suffix,
        const std::string &stop);
  ~Llama();

  bool embedding;

  std::string detokenize(const std::vector<llama_token> &tokens);
  std::vector<llama_token> tokenize(const std::string &text, bool add_bos);
  void reset();
  void cancel();
  std::vector<float> create_embeddings(const std::string &input_prompt);
  std::string generate_response(const std::string &input_prompt,
                                bool add_pfx_sfx = true,
                                const llama_sampling_params &sampling_params =
                                    llama_sampling_default_params(),
                                GenerateResponseCallback callbakc = nullptr);

protected:
  llama_model *model;
  llama_context *ctx;
  void eval();
  llama_token sample(llama_sampling_params sampling_params);

private:
  Spinner spinner;

  int32_t n_ctx; // context size
  llama_eval_params eval_params;

  // prefix, suffix, stop
  std::string stop;
  std::vector<llama_token> inp_pfx;
  std::vector<llama_token> inp_sfx;

  // aux
  std::vector<llama_token> last_n_tokens;
  std::vector<llama_token> prompt_tokens;
  std::vector<llama_token> batch_tokens;

  bool is_antiprompt;
  bool canceled;
  int32_t n_past;
  int32_t n_remain;
  int32_t n_consumed;
};

} // namespace llama_ros

#endif
