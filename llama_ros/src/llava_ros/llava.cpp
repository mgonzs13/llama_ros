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

#include <cassert>
#include <cmath>
#include <thread>

#include "base64.hpp"
#include "common.h"
#include "llava_ros/llava.hpp"

using namespace llava_ros;

Llava::Llava(rclcpp::Logger logger, std::shared_ptr<struct gpt_params> params,
             bool debug)
    : llama_ros::Llama(logger, params, debug) {

  // load clip model
  const char *clip_path = this->params->mmproj.c_str();
  auto ctx_clip = clip_model_load(clip_path, 1);
  this->image_embed = nullptr;

  // create llava ctx
  this->ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

  this->ctx_llava->ctx_llama = this->ctx;
  this->ctx_llava->ctx_clip = ctx_clip;
  this->ctx_llava->model = this->model;
}

Llava::~Llava() {
  if (this->ctx_llava->ctx_clip) {
    clip_free(this->ctx_llava->ctx_clip);
    this->ctx_llava->ctx_clip = NULL;
  }

  this->free_image();

  this->llama_ros::Llama::~Llama();

  free(this->ctx_llava);
}

void Llava::free_image() {
  if (this->image_embed != nullptr) {
    llava_image_embed_free(this->image_embed);
    this->image_embed = nullptr;
  }
}

bool Llava::load_image(std::string base64_str) {

  this->free_image();

  this->image_embed = this->base64_image_to_embed(base64_str);

  if (this->image_embed == nullptr) {
    RCLCPP_ERROR(this->logger, "Can't load base64 image");
    return false;
  }

  return true;
}

struct llava_image_embed *
Llava::base64_image_to_embed(const std::string &base64_str) {

  auto required_bytes = base64::required_encode_size(base64_str.size());
  auto img_bytes = std::vector<unsigned char>(required_bytes);
  base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

  auto embed = llava_image_embed_make_with_bytes(
      this->ctx_llava->ctx_clip, this->params->n_threads, img_bytes.data(),
      img_bytes.size());

  if (!embed) {
    RCLCPP_ERROR(this->logger, "Could not load image from base64 string");
    return nullptr;
  }

  return embed;
}

bool Llava::eval_image() {

  int n_embd = this->get_n_embd();
  bool succ = true;

  for (int i = 0; i < this->image_embed->n_image_pos;
       i += this->params->n_batch) {

    int n_eval = this->image_embed->n_image_pos - i;

    if (n_eval > this->params->n_batch) {
      n_eval = this->params->n_batch;
    }

    llama_batch batch = {
        int32_t(n_eval),
        nullptr,
        (this->image_embed->embed + i * n_embd),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        this->n_past,
        1,
        0,
    };

    if (this->debug) {
      this->spinner.spin("EVALUATING IMAGE " + std::to_string(n_eval) +
                         " TOKENS");
    }

    if (llama_decode(this->ctx_llava->ctx_llama, batch)) {
      RCLCPP_ERROR(this->logger, "Failed in image eval");
      succ = false;
      break;
    }

    this->n_past += n_eval;
  }

  this->free_image();
  return succ;
}

bool Llava::eval_prompt() {

  // check if there is a prefix in the prompt_tokens
  // only if there is an image to eval
  std::vector<llama_token> inp_pfx = this->tokenize(
      this->params->input_prefix,
      this->should_add_bos_token() && !this->prompt_tokens.size(), true);

  bool is_prefix = true;

  if (inp_pfx.size() > this->prompt_tokens.size() - this->n_consumed &&
      this->image_embed != nullptr) {
    is_prefix = false;

  } else {
    int j = this->n_consumed;
    for (size_t i = 0; i < inp_pfx.size(); i++) {

      if (inp_pfx[i] != this->prompt_tokens[j]) {
        is_prefix = false;
        break;
      }

      j++;
    }
  }

  // eval the prefix before the image
  if (is_prefix) {

    for (size_t i = 0; i < inp_pfx.size(); i++) {
      this->batch_tokens.push_back(inp_pfx[i]);
      this->n_consumed++;
    }

    if (!this->eval()) {
      return false;
    }
  }

  // eval the image
  if (this->image_embed != nullptr) {
    RCLCPP_INFO(this->logger, "Evaluating the image");

    if (!this->eval_image()) {
      return false;
    }
  }

  // eval the rest of the prompt
  return llama_ros::Llama::eval_prompt();
}
