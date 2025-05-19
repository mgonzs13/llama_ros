// MIT License
//
// Copyright (c) 2024 Miguel Ángel González Santamarta
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

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <cassert>
#include <cmath>

#include "base64.hpp"
#include "common.h"

#include "llama_utils/logs.hpp"
#include "llava_ros/llava.hpp"

using namespace llava_ros;

Llava::Llava(const struct common_params &params, std::string system_prompt)
    : llama_ros::Llama(params, system_prompt, false), image_chunk(nullptr),
      chunks(mtmd_input_chunks_init()), image_pose(0) {

  // create mtmd params
  mtmd_context_params mparams = mtmd_context_params_default();
  mparams.use_gpu = params.mmproj_use_gpu;
  mparams.print_timings = false;
  mparams.n_threads = params.cpuparams.n_threads;
  mparams.verbosity =
      params.verbosity > 0 ? GGML_LOG_LEVEL_DEBUG : GGML_LOG_LEVEL_INFO;

  // load multimodal model
  this->mtmd_ctx = mtmd_init_from_file(this->params.mmproj.path.c_str(),
                                       this->get_model(), mparams);

  // set inital values
  this->reset();
}

Llava::~Llava() { mtmd_free(this->mtmd_ctx); }

void Llava::reset() {
  this->image_chunk = nullptr;
  Llama::reset();
}

/*
*****************************
*        LOAD IMAGE         *
*****************************
*/
void Llava::load_prompt(const std::string &input_prompt, bool add_pfx,
                        bool add_sfx) {

  mtmd_input_text inp_txt = {
      input_prompt.c_str(),
      /* add_special */ true,
      /* parse_special */ true,
  };

  auto bitmaps_c_ptr = this->bitmaps.c_ptr();

  if (mtmd_tokenize(this->mtmd_ctx, this->chunks.ptr.get(), &inp_txt,
                    bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
    LLAMA_LOG_ERROR("Failed to tokenize prompt");
    return;
  }

  // calculate the image pose
  this->image_chunk = nullptr;
  this->image_pose = -1;
  int aux_pose = 0;

  // insert prefix
  if (add_pfx) {
    this->load_prefix();
  }

  for (size_t i = 0; i < this->chunks.size(); i++) {

    if (mtmd_input_chunk_get_type(this->chunks[i]) ==
        MTMD_INPUT_CHUNK_TYPE_IMAGE) {

      this->image_chunk = this->chunks[i];

      if (this->image_pose == -1) {
        this->image_pose = aux_pose;
      }

    } else if (mtmd_input_chunk_get_type(this->chunks[i]) ==
               MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n_tokens;
      aux_pose += n_tokens;
      auto tokens =
          mtmd_input_chunk_get_tokens_text(this->chunks[i], &n_tokens);

      for (size_t j = 0; j < n_tokens; j++) {
        this->prompt_tokens.push_back(tokens[j]);
      }
    }
  }

  // insert suffix
  if (add_sfx) {
    this->load_suffix();
  }
}

// Computes FNV-1a hash of the data
static std::string fnv_hash(const uint8_t *data, size_t len) {
  const uint64_t fnv_prime = 0x100000001b3ULL;
  uint64_t hash = 0xcbf29ce484222325ULL;

  for (size_t i = 0; i < len; ++i) {
    hash ^= data[i];
    hash *= fnv_prime;
  }
  return std::to_string(hash);
}

bool Llava::load_image(std::vector<uint8_t> buf) {

  LLAMA_LOG_INFO("Loading image...");

  mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_buf(buf.data(), buf.size()));

  if (!bmp.ptr) {
    LLAMA_LOG_ERROR("Can't load image");
    return false;
  }

  // calculate bitmap hash (for KV caching)
  std::string hash = fnv_hash(bmp.data(), bmp.nx() * bmp.ny() * 3);
  bmp.set_id(hash.c_str());
  this->bitmaps.entries.clear();
  this->bitmaps.entries.push_back(std::move(bmp));

  return true;
}

/*
*****************************
*        EVAL IMAGE         *
*****************************
*/
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
    if (this->image_chunk != nullptr) {
      LLAMA_LOG_INFO("Evaluating the image");

      if (!this->eval_image()) {
        LLAMA_LOG_ERROR("Error evaluating the image");
        return false;
      }

    } else {
      LLAMA_LOG_INFO("No image to evaluate");
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

bool Llava::eval_image() {
  return mtmd_helper_eval_chunk_single(this->mtmd_ctx, this->ctx, // contexts
                                       this->image_chunk,         // image chunk
                                       this->n_past,              // n_past
                                       0,                         // seq_id
                                       this->params.n_batch,      // batch
                                       true,                      // logits last
                                       &this->n_past) == 0;
}