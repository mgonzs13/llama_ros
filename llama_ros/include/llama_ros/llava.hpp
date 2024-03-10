// MIT License

// Copyright (c) 2024  Miguel Ángel González Santamarta

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

#ifndef LLAMA_ROS__LLAVA_HPP
#define LLAMA_ROS__LLAVA_HPP

#include <functional>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>

#include "clip.h"
#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "llava.h"

namespace llama_ros {

struct llava_context {
  struct clip_ctx *ctx_clip = NULL;
  struct llama_context *ctx_llama = NULL;
  struct llama_model *model = NULL;
};

class Llava {

  using GenerateResponseCallback = std::function<void(std::string)>;

public:
  Llava(rclcpp::Logger logger, std::shared_ptr<struct gpt_params> params,
        bool debug);
  ~Llava();

  struct llava_context *llava_init(const gpt_params &params);
  void llava_free(struct llava_context *ctx_llava);

  const struct llava_context *get_ctx() { return this->ctx_llava; }
  const struct llama_model *get_llama_model() {
    return llama_get_model(this->ctx_llava->ctx_llama);
  }
  int get_n_ctx() { return llama_n_ctx(this->ctx_llava->ctx_llama); }
  int get_n_ctx_train() { return llama_n_ctx_train(this->get_llama_model()); }
  int get_n_embd() { return llama_n_embd(this->get_llama_model()); }
  int get_n_vocab() { return llama_n_vocab(this->get_llama_model()); }
  bool is_embedding() { return this->params->embedding; }
  bool should_add_bos_token() {
    return llama_should_add_bos_token(this->get_llama_model());
  }
  llama_token get_token_eos() {
    return llama_token_eos(this->get_llama_model());
  }

  struct llava_image_embed *load_image(std::string base64_str);
  struct llava_image_embed *
  base64_image_to_embed(const std::string &base64_str);

  void cancel();
  std::string process_prompt(struct llava_image_embed *image_embed,
                             const std::string &prompt,
                             GenerateResponseCallback callback);

protected:
  struct llava_context *ctx_llava;

  bool eval_tokens(std::vector<llama_token> tokens, int n_batch, int *n_past);
  bool eval_id(int id, int *n_past);
  bool eval_string(const char *str, int n_batch, int *n_past, bool add_bos);
  const char *sample(struct llama_sampling_context *ctx_sampling, int *n_past);

private:
  rclcpp::Logger logger;
  std::shared_ptr<struct gpt_params> params;
  bool debug;
  bool canceled;
};

} // namespace llama_ros

#endif
