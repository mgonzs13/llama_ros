// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
// Copyright (c) 2026 Alejandro González Cantón
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
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "llava_ros/llava_request_handler.hpp"
#include "llama_utils/chat_utils.hpp"
#include "llama_utils/logs.hpp"
#include "llava_ros/llava.hpp"

using namespace llava_ros;

LlavaCompletionRequestHandler::LlavaCompletionRequestHandler(Llava *llava)
    : RequestHandler(llava), llava_(llava) {}

void LlavaCompletionRequestHandler::handle(
    const std::string &input_prompt, llama_ros::ServerSlot *slot,
    common_params_sampling sparams,
    std::function<void(llama_ros::CompletionOutput, llama_ros::ServerSlot *)>
        callback,
    std::vector<std::string> stop, bool reset) {
  (void)reset; // Unused parameter

  LLAMA_LOG_INFO("Handling completion request with prompt: %s",
                 input_prompt.c_str());

  slot->prompt_tokens.clear();
  std::string converted_prompt = input_prompt;

  if (this->llava_->params.input_prefix.size() > 0) {
    converted_prompt.insert(0, this->llava_->params.input_prefix);
  }

  if (this->llava_->params.input_suffix.size() > 0) {
    converted_prompt.append(this->llava_->params.input_suffix.c_str());
  }

  std::string prompt_str = converted_prompt;
  mtmd_input_text inp_txt = {
      prompt_str.c_str(),
      /* add_special */ true,
      /* parse_special */ true,
  };
  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  auto bitmaps_c_ptr = this->llava_->bitmaps.c_ptr();
  int32_t tokenized =
      mtmd_tokenize(this->llava_->mtmd_ctx, chunks.ptr.get(), &inp_txt,
                    bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
  if (tokenized != 0) {
    throw std::runtime_error("Failed to tokenize prompt");
  }

  this->llava_->process_input_chunks(chunks, slot);

  LLAMA_LOG_INFO("Tokenized prompt to %ld tokens", slot->prompt_tokens.size());

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->params.sampling = sparams;
  slot->sampler =
      common_sampler_init(this->llava_->model, slot->params.sampling);
  slot->stream_callback = callback;
  slot->params.antiprompt.insert(slot->params.antiprompt.end(), stop.begin(),
                                 stop.end());
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = llama_ros::SERVER_TASK_TYPE_COMPLETION;
  slot->state = llama_ros::SLOT_STATE_STARTED;
}

LlavaChatCompletionRequestHandler::LlavaChatCompletionRequestHandler(
    Llava *llava)
    : RequestHandler(llava), llava_(llava) {}

void LlavaChatCompletionRequestHandler::handle(
    llama_utils::ChatCompletionsContext chat_context,
    llama_ros::ServerSlot *slot,
    std::function<void(llama_ros::CompletionOutput, llama_ros::ServerSlot *)>
        callback) {
  LLAMA_LOG_INFO("Handling chat completion request");

  common_chat_templates_inputs inputs = chat_context.prompt_format_config;
  slot->params.oaicompat_chat_syntax = chat_context.oaicompat_chat_syntax;
  slot->params.sampling = chat_context.sparams;

  std::string prompt_str = chat_context.chat_prompt_instance.prompt;
  mtmd_input_text inp_txt = {
      prompt_str.c_str(),
      /* add_special */ true,
      /* parse_special */ true,
  };
  mtmd::input_chunks chunks(mtmd_input_chunks_init());
  auto bitmaps_c_ptr = this->llava_->bitmaps.c_ptr();
  int32_t tokenized =
      mtmd_tokenize(this->llava_->mtmd_ctx, chunks.ptr.get(), &inp_txt,
                    bitmaps_c_ptr.data(), bitmaps_c_ptr.size());
  if (tokenized != 0) {
    throw std::runtime_error("Failed to tokenize prompt");
  }

  this->llava_->process_input_chunks(chunks, slot);

  LLAMA_LOG_INFO("Tokenized prompt to %ld tokens", slot->prompt_tokens.size());

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->sampler =
      common_sampler_init(this->llava_->model, slot->params.sampling);
  slot->stream_callback = callback;
  slot->chat_format = chat_context.chat_prompt_instance.format;
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = llama_ros::SERVER_TASK_TYPE_COMPLETION;
  slot->state = llama_ros::SLOT_STATE_STARTED;
}
