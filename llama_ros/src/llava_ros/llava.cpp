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

#include "llama_ros/llama.hpp"
#include "llama_utils/logs.hpp"
#include "llava_ros/llava.hpp"

using namespace llava_ros;

Llava::Llava(const struct common_params &params, std::string system_prompt)
    : llama_ros::Llama(params, system_prompt, false),
      chunks(mtmd_input_chunks_init()) {

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

void Llava::reset() { Llama::reset(); }

/*
*****************************
*        LOAD MTMDS         *
*****************************
*/
void Llava::load_prompt(const std::string &input_prompt, bool add_pfx,
                        bool add_sfx, llama_ros::ServerSlot *slot) {

  std::string prompt = input_prompt;

  if (add_pfx && !this->check_if_prefix(slot)) {
    prompt = this->params.input_prefix + prompt;
  }

  if (add_sfx) {
    prompt += this->params.input_suffix;
  }

  mtmd_input_text inp_txt = {
      prompt.c_str(),
      /* add_special */ true,
      /* parse_special */ true,
  };

  auto bitmaps_c_ptr = this->bitmaps.c_ptr();

  if (mtmd_tokenize(this->mtmd_ctx, this->chunks.ptr.get(), &inp_txt,
                    bitmaps_c_ptr.data(), bitmaps_c_ptr.size()) != 0) {
    LLAMA_LOG_ERROR("Failed to tokenize prompt");
    return;
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

bool Llava::load_mtmd(std::vector<uint8_t> buf) {

  LLAMA_LOG_INFO("Loading mtmd...");

  mtmd::bitmap bmp(
      mtmd_helper_bitmap_init_from_buf(this->mtmd_ctx, buf.data(), buf.size()));

  if (!bmp.ptr) {
    LLAMA_LOG_ERROR("Can't load mtmd");
    return false;
  }

  // calculate bitmap hash (for KV caching)
  std::string hash = fnv_hash(bmp.data(), bmp.nx() * bmp.ny() * 3);
  bmp.set_id(hash.c_str());
  this->bitmaps.entries.push_back(std::move(bmp));

  return true;
}

bool Llava::load_mtmds(std::vector<std::vector<uint8_t>> mtmds) {

  LLAMA_LOG_INFO("Loading mtmds...");

  for (const auto &mtmd : mtmds) {
    if (!this->load_mtmd(mtmd)) {
      LLAMA_LOG_ERROR("Failed to load mtmd");
      return false;
    }
  }

  return true;
}

void Llava::clear_mtmds() {
  LLAMA_LOG_ERROR("Clearing mtmds...");
  this->bitmaps.entries.clear();
}

/*
*****************************
*        EVAL IMAGE         *
*****************************
*/
bool Llava::eval_prompt(llama_ros::ServerSlot *slot) {

  for (size_t i = 0; i < this->chunks.size(); i++) {

    auto chunk = mtmd_input_chunk_copy(this->chunks[i]);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_IMAGE ||
        chunk_type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {

      LLAMA_LOG_INFO("Evaluating mtmd data");

      if (!this->eval_mtmd_chunk(chunk)) {
        LLAMA_LOG_ERROR("Error evaluating the image");
        return false;
      }

    } else if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {

      LLAMA_LOG_INFO("Evaluating text");

      size_t n_tokens;
      auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);

      for (size_t j = 0; j < n_tokens; j++) {
        slot->prompt_tokens.push_back(tokens[j]);
      }

      if (!Llama::eval_prompt(slot->prompt_tokens, slot)) {
        return false;
      }
    }

    mtmd_input_chunk_free(chunk);
  }

  LLAMA_LOG_INFO("llava prompt: %s",
                 this->detokenize(slot->prompt_tokens).c_str());

  return true;
}

bool Llava::eval_mtmd_chunk(const mtmd_input_chunk *image_chunk) {
  return mtmd_helper_eval_chunk_single(this->mtmd_ctx, this->ctx, // contexts
                                       image_chunk,               // image chunk
                                       this->n_past,              // n_past
                                       0,                         // seq_id
                                       this->params.n_batch,      // batch
                                       true,                      // logits last
                                       &this->n_past) == 0;
}

