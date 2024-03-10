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

public:
  Llava(rclcpp::Logger logger, const struct gpt_params &params, bool debug);
  ~Llava();

  struct llava_context *llava_init(const gpt_params &params);
  void llava_free(struct llava_context *ctx_llava);

  void cancel();

  struct llava_image_embed *load_image(std::string base64_str);
  struct llava_image_embed *
  base64_image_to_embed(const std::string &base64_str);

  std::string process_prompt(struct llava_image_embed *image_embed,
                             const std::string &prompt);

protected:
  struct llava_context *ctx_llava;

  bool eval_tokens(std::vector<llama_token> tokens, int n_batch, int *n_past);
  bool eval_id(int id, int *n_past);
  bool eval_string(const char *str, int n_batch, int *n_past, bool add_bos);
  const char *sample(struct llama_sampling_context *ctx_sampling, int *n_past);

private:
  rclcpp::Logger logger;
  struct gpt_params params;
  bool debug;
  bool canceled;
};

} // namespace llama_ros

#endif
