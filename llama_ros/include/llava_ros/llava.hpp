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
#include <string>
#include <vector>

#include "clip.h"
#include "common.h"
#include "llava.h"

#include "llama_ros/llama.hpp"

namespace llava_ros {

struct LlavaParams {
  std::string image_text = "";
  std::string image_prefix = "";
  std::string image_suffix = "";
};

class Llava : public llama_ros::Llama {

public:
  Llava(const struct common_params &params,
        const struct LlavaParams &llava_params, std::string system_prompt = "");
  ~Llava();

  bool load_image(std::string base64_str);
  struct llava_image_embed *
  base64_image_to_embed(const std::string &base64_str);

protected:
  void load_prompt(const std::string &input_prompt, bool add_pfx,
                   bool add_sfx) override;
  bool eval_image(struct llava_image_embed *image_embed);
  bool eval_prompt();

  struct llava_image_embed *image_embed;
  struct clip_ctx *ctx_clip;
  struct LlavaParams llava_params;

private:
  void free_image();
  int image_pose;
};

} // namespace llava_ros

#endif
