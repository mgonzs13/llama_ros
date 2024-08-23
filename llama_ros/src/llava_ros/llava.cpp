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

Llava::Llava(const struct gpt_params &params,
             const struct llava_params &llava_params, bool debug)
    : llama_ros::Llama(params, debug), llava_params(llava_params) {

  // load clip model
  const char *clip_path = this->params.mmproj.c_str();
  this->ctx_clip = clip_model_load(clip_path, 1);
  this->image_embed = nullptr;
}

Llava::~Llava() {
  clip_free(this->ctx_clip);
  this->free_image();
}

/*
*****************************
*        LOAD IMAGE         *
*****************************
*/
void Llava::load_prompt(const std::string &input_prompt, bool add_pfx,
                        bool add_sfx) {

  std::string prompt(input_prompt);
  this->image_pose = -1;

  // image
  if (this->image_embed != nullptr) {

    // search for image_text
    size_t image_pos = prompt.find(this->llava_params.image_text);

    // empty prompt
    if (prompt.size() == 0) {
      prompt = this->llava_params.image_text;

    } else if (prompt.size() > 0) {

      // no image_text
      if (image_pos == std::string::npos) {
        prompt = this->llava_params.image_text + "\n" + prompt;
        image_pos = 0;
      }
    }

    // split prompt
    std::string prompt_1 =
        prompt.substr(0, image_pos) + this->llava_params.image_prefix;
    std::string prompt_2 =
        this->llava_params.image_suffix +
        prompt.substr(image_pos + this->llava_params.image_text.length());

    // load first part of the prompt
    Llama::load_prompt(prompt_1, true, false);

    // get image pose
    this->image_pose = (int)this->prompt_tokens.size();

    // load second part of the prompt
    Llama::load_prompt(prompt_2, false, true);

    // no image
  } else if (this->image_embed == nullptr) {
    Llama::load_prompt(input_prompt, add_pfx, add_sfx);
  }
}

bool Llava::load_image(std::string base64_str) {

  this->free_image();

  LLAMA_LOG_INFO("Converting base64 image to embeddings");
  this->image_embed = this->base64_image_to_embed(base64_str);

  if (this->image_embed == nullptr) {
    LLAMA_LOG_ERROR("Can't load base64 image");
    return false;
  }

  return true;
}

void Llava::free_image() {
  if (this->image_embed != nullptr) {
    llava_image_embed_free(this->image_embed);
    this->image_embed = nullptr;
  }
}

struct llava_image_embed *
Llava::base64_image_to_embed(const std::string &base64_str) {

  auto required_bytes = base64::required_encode_size(base64_str.size());
  auto img_bytes = std::vector<unsigned char>(required_bytes);
  base64::decode(base64_str.begin(), base64_str.end(), img_bytes.begin());

  auto embed =
      llava_image_embed_make_with_bytes(this->ctx_clip, this->params.n_threads,
                                        img_bytes.data(), img_bytes.size());

  if (!embed) {
    LLAMA_LOG_ERROR("Could not load image from base64 string");
    return nullptr;
  }

  return embed;
}

/*
*****************************
*        EVAL IMAGE         *
*****************************
*/
bool Llava::eval_image(struct llava_image_embed *image_embed) {

  int n_embd = this->get_n_embd();
  bool succ = true;

  for (int i = 0; i < image_embed->n_image_pos; i += this->params.n_batch) {

    int n_eval = image_embed->n_image_pos - i;

    if (n_eval > this->params.n_batch) {
      n_eval = this->params.n_batch;
    }

    llama_batch batch = {
        int32_t(n_eval),
        nullptr,
        (image_embed->embed + i * n_embd),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        this->n_past,
        1,
        0,
    };

    if (!this->eval(batch)) {
      LLAMA_LOG_ERROR("Failed in image eval");
      succ = false;
      break;
    }
  }

  this->free_image();
  return succ;
}

bool Llava::eval_prompt() {

  if (this->image_pose >= 0) {

    std::vector<llama_token> prompt_tokens_1(this->prompt_tokens.begin(),
                                             this->prompt_tokens.begin() +
                                                 this->image_pose);

    // eval part of the prompt
    if (!Llama::eval_prompt(prompt_tokens_1)) {
      return false;
    }

    // eval the image
    if (this->image_embed != nullptr) {
      LLAMA_LOG_INFO("Evaluating the image");

      if (!this->eval_image(this->image_embed)) {
        return false;
      }
    }

    // eval the full prompt
    if (!Llama::eval_prompt(this->prompt_tokens)) {
      return false;
    }

  } else { // no image in the prompt
    return llama_ros::Llama::eval_prompt();
  }

  return true;
}
