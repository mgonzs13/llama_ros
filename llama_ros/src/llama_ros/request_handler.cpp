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

#include "llama_ros/request_handler.hpp"
#include "llama_ros/llama.hpp"
#include "llama_utils/chat_utils.hpp"
#include "llama_utils/logs.hpp"

using namespace llama_ros;

// EmbeddingRequestHandler implementation
std::vector<llama_token>
EmbeddingRequestHandler::truncate_tokens(const std::vector<llama_token> &tokens,
                                         int limit_size, bool add_eos) {
  std::vector<llama_token> new_tokens = tokens;

  // Reserve space for EOS token if needed
  int effective_limit = limit_size;
  if (add_eos && (tokens.empty() || tokens.back() != llama_->get_token_eos())) {
    effective_limit = limit_size - 1;
  }

  if ((int)tokens.size() > effective_limit) {
    LLAMA_LOG_WARN("Prompt too long %ld, limit size %d, truncating...",
                   tokens.size(), limit_size);
    new_tokens.resize(effective_limit);
  }

  if (add_eos && !new_tokens.empty() &&
      new_tokens.back() != llama_->get_token_eos()) {
    new_tokens.push_back(llama_->get_token_eos());
  }

  return new_tokens;
}

void EmbeddingRequestHandler::handle(const std::string &input_prompt,
                                     ServerSlot *slot) {
  auto tokens = common_tokenize(llama_->get_ctx(), input_prompt, false, true);
  LLAMA_LOG_INFO("Tokenized prompt to %ld tokens", tokens.size());
  tokens = truncate_tokens(tokens, llama_n_batch(llama_->get_ctx()), true);

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->sampler =
      common_sampler_init(llama_->get_model(), llama_->params.sampling);
  slot->prompt_tokens = tokens;
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = SERVER_TASK_TYPE_EMBEDDING;
  slot->state = SLOT_STATE_STARTED;
}

// RerankRequestHandler implementation
std::vector<llama_token>
RerankRequestHandler::truncate_tokens(const std::vector<llama_token> &tokens,
                                      int limit_size, bool add_eos) {
  std::vector<llama_token> new_tokens = tokens;

  // Reserve space for EOS token if needed
  int effective_limit = limit_size;
  if (add_eos && (tokens.empty() || tokens.back() != llama_->get_token_eos())) {
    effective_limit = limit_size - 1;
  }

  if ((int)tokens.size() > effective_limit) {
    LLAMA_LOG_WARN("Prompt too long %ld, limit size %d, truncating...",
                   tokens.size(), limit_size);
    new_tokens.resize(effective_limit);
  }

  if (add_eos && !new_tokens.empty() &&
      new_tokens.back() != llama_->get_token_eos()) {
    new_tokens.push_back(llama_->get_token_eos());
  }

  return new_tokens;
}

void RerankRequestHandler::handle(const std::string &query,
                                  const std::string &document,
                                  ServerSlot *slot) {
  std::vector<llama_token> tokens;
  tokens.push_back(llama_->get_token_bos());

  auto tokens_query = common_tokenize(llama_->get_vocab(), query, false, true);
  auto truncated_query = truncate_tokens(
      tokens_query, (int)(llama_->params.n_batch / 2) - 2, true);
  tokens.insert(tokens.end(), truncated_query.begin(), truncated_query.end());
  tokens.push_back(llama_->get_token_eos());
  tokens.push_back(llama_->get_token_sep());

  auto tokens_document =
      common_tokenize(llama_->get_vocab(), document, false, true);
  auto truncated_document = truncate_tokens(
      tokens_document, (int)(llama_->params.n_batch / 2) - 2, true);
  tokens.insert(tokens.end(), truncated_document.begin(),
                truncated_document.end());
  tokens.push_back(llama_->get_token_eos());

  tokens = truncate_tokens(tokens, llama_n_batch(llama_->get_ctx()), true);

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->sampler =
      common_sampler_init(llama_->get_model(), llama_->params.sampling);
  slot->prompt_tokens = tokens;
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = SERVER_TASK_TYPE_RERANK;
  slot->state = SLOT_STATE_STARTED;
}

// CompletionRequestHandler implementation
void CompletionRequestHandler::handle(
    const std::string &input_prompt, ServerSlot *slot,
    common_params_sampling sparams,
    std::function<void(CompletionOutput, ServerSlot *)> callback,
    std::vector<std::string> stop, bool /*reset*/) {
  slot->prompt_tokens.clear();
  std::string full_prompt = "";

  if (llama_->params.input_prefix.size() > 0) {
    full_prompt += llama_->params.input_prefix;
  }
  full_prompt += input_prompt;
  if (llama_->params.input_suffix.size() > 0) {
    full_prompt += llama_->params.input_suffix;
  }

  slot->prompt_tokens =
      common_tokenize(llama_->get_ctx(), full_prompt, false, true);
  LLAMA_LOG_INFO("Full prompt: '%s'", full_prompt.c_str());
  LLAMA_LOG_INFO("Tokenized prompt to %ld tokens", slot->prompt_tokens.size());

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->params.sampling = sparams;
  slot->sampler =
      common_sampler_init(llama_->get_model(), llama_->params.sampling);
  slot->stream_callback = callback;
  slot->params.antiprompt.insert(slot->params.antiprompt.end(), stop.begin(),
                                 stop.end());
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = SERVER_TASK_TYPE_COMPLETION;
  slot->state = SLOT_STATE_STARTED;
}

// ChatCompletionRequestHandler implementation
void ChatCompletionRequestHandler::handle(
    llama_utils::ChatCompletionsContext chat_context, ServerSlot *slot,
    std::function<void(CompletionOutput, ServerSlot *)> callback) {
  // No longer need to get chat_tmpls - already initialized in ChatFormatter

  slot->reset();

  common_chat_templates_inputs inputs = chat_context.prompt_format_config;
  slot->params.oaicompat_chat_syntax = chat_context.oaicompat_chat_syntax;
  slot->params.sampling = chat_context.sparams;

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->prompt_tokens = common_tokenize(
      llama_->get_ctx(), chat_context.chat_prompt_instance.prompt, false, true);
  slot->sampler =
      common_sampler_init(llama_->get_model(), slot->params.sampling);
  slot->stream_callback = callback;
  slot->chat_format = chat_context.chat_prompt_instance.format;
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = SERVER_TASK_TYPE_COMPLETION;
  slot->state = SLOT_STATE_STARTED;
}
