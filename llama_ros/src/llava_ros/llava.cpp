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
#include "llama_utils/chat_utils.hpp"
#include "llava_ros/llava.hpp"

using namespace llava_ros;

Llava::Llava(const struct common_params &params, std::string system_prompt)
    : llama_ros::Llama(params, system_prompt, false) {

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

const mtmd::input_chunk_ptr & find_chunk(llama_pos pos, llama_ros::ServerSlot * slot) {
    auto it = slot->map_pos_to_media.find(pos);
    if (it != slot->map_pos_to_media.end()) {
        return it->second;
    } else {
        throw std::runtime_error("Chunk not found");
    }
}

int32_t process_chunk(
            llama_context * ctx,
            mtmd_context * mctx,
            llama_pos n_past,
            int32_t seq_id,
            llama_pos & n_pos_out,
            llama_ros::ServerSlot * slot) {
    auto & chunk = find_chunk(n_past, slot);
    const char * name = mtmd_input_chunk_get_type(chunk.get()) == MTMD_INPUT_CHUNK_TYPE_IMAGE
                        ? "image" : "audio";
    LLAMA_LOG_INFO("processing %s...\n", name);
    int32_t n_batch = llama_n_batch(ctx);
    llama_pos new_n_past = n_past;
    int32_t result = mtmd_helper_eval_chunk_single(mctx, ctx,
        chunk.get(),
        n_past,
        seq_id,
        n_batch,
        true, // logits last
        &new_n_past);
    if (result != 0) {
        n_pos_out = n_past;
        return result;
    }
    n_pos_out = new_n_past;
    return 0;
}


bool Llava::process_mtmd_chunk(llama_ros::ServerSlot *slot) {
  if (!slot) {
    return false;
  }

  int32_t new_n_past;
  int32_t res = process_chunk(this->ctx, this->mtmd_ctx, slot->n_past, slot->id, new_n_past, slot);
  int32_t n_pos = new_n_past - slot->n_past;

  if (res != 0) {
      LLAMA_LOG_ERROR("failed to process image, res = %d\n", res);
      return false;
  }

  slot->n_past                    += n_pos;
  slot->n_prompt_tokens_processed += n_pos;
  return true;
}

void Llava::process_input_chunks(mtmd::input_chunks &chunks, llama_ros::ServerSlot *slot) {
  for (size_t i = 0; i < chunks.size(); i++) {
    auto chunk = mtmd_input_chunk_copy(chunks[i]);
    auto chunk_type = mtmd_input_chunk_get_type(chunk);

    if (chunk_type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
      size_t n_tokens;
      auto tokens = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);

      for (size_t j = 0; j < n_tokens; j++) {
        slot->prompt_tokens.push_back(tokens[j]);
      }
    } else {
      const int n_pos = mtmd_input_chunk_get_n_pos(chunk);
      llama_pos start_pos = slot->prompt_tokens.size();
      for (int i = 0; i < n_pos; ++i) {
        slot->prompt_tokens.emplace_back(LLAMA_TOKEN_NULL);
      }
      mtmd::input_chunk_ptr new_chunk(mtmd_input_chunk_copy(chunk));
      slot->map_pos_to_media[start_pos] = std::move(new_chunk);
    }
  }
}

void llava_ros::Llava::handle_completion_req(
      const std::string &input_prompt, llama_ros::ServerSlot *slot,
      struct common_params_sampling sparams,
      llama_ros::ServerSlot::GenerateResponseCallback callback,
      std::vector<std::string> stop, bool reset) {
  slot->prompt_tokens.clear();
  std::string converted_prompt = input_prompt;

  if (this->params.input_prefix.size() > 0) {
    converted_prompt.insert(0, this->params.input_prefix);
  }
  if (this->params.input_suffix.size() > 0) {
    converted_prompt.append(this->params.input_suffix.c_str());
  }

  std::string prompt_str = converted_prompt;
  mtmd_input_text inp_txt = {
      prompt_str.c_str(),
      /* add_special */   true,
      /* parse_special */ true,
  };
  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  auto bitmaps_c_ptr = bitmaps.c_ptr();
  int32_t tokenized = mtmd_tokenize(this->mtmd_ctx,
                                      chunks.ptr.get(),
                                      &inp_txt,
                                      bitmaps_c_ptr.data(),
                                      bitmaps_c_ptr.size());
  if (tokenized != 0) {
      throw std::runtime_error("Failed to tokenize prompt");
  }

  process_input_chunks(chunks, slot);

  LLAMA_LOG_INFO("Tokenized prompt to %ld tokens", slot->prompt_tokens.size());
  
  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->params.sampling = sparams;
  slot->sampler = common_sampler_init(model, params.sampling);
  slot->stream_callback = callback;
  slot->params.antiprompt.insert(slot->params.antiprompt.end(), stop.begin(),
                                 stop.end());
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = llama_ros::SERVER_TASK_TYPE_COMPLETION;
  slot->state = llama_ros::SLOT_STATE_STARTED;
}

void llava_ros::Llava::handle_chat_completion_req(
      llama_utils::ChatCompletionsContext chat_context, llama_ros::ServerSlot *slot,
      llama_ros::ServerSlot::GenerateResponseCallback callback) {
    auto chat_tmpls = this->get_chat_templates();
  common_chat_templates_inputs inputs = chat_context.prompt_format_config;
  slot->params.oaicompat_chat_syntax = chat_context.oaicompat_chat_syntax;
  slot->params.sampling = chat_context.sparams;

  std::string prompt_str = chat_context.chat_prompt_instance.prompt;
  mtmd_input_text inp_txt = {
      prompt_str.c_str(),
      /* add_special */   true,
      /* parse_special */ true,
  };
  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  auto bitmaps_c_ptr = bitmaps.c_ptr();
  int32_t tokenized = mtmd_tokenize(this->mtmd_ctx,
                                      chunks.ptr.get(),
                                      &inp_txt,
                                      bitmaps_c_ptr.data(),
                                      bitmaps_c_ptr.size());
  if (tokenized != 0) {
      throw std::runtime_error("Failed to tokenize prompt");
  }

  process_input_chunks(chunks, slot);

  LLAMA_LOG_INFO("Tokenized prompt to %ld tokens", slot->prompt_tokens.size());

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->sampler = common_sampler_init(model, slot->params.sampling);
  slot->stream_callback = callback;
  slot->chat_format = chat_context.chat_prompt_instance.format;
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = llama_ros::SERVER_TASK_TYPE_COMPLETION;
  slot->state = llama_ros::SLOT_STATE_STARTED;
}

