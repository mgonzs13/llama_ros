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

  // Vocabs without an EOS token (BERT-style embedding models) must never get
  // LLAMA_TOKEN_NULL appended: it would be mistaken for a media placeholder
  const llama_token eos = this->llama_->get_token_eos();
  const bool append_eos = add_eos && eos != LLAMA_TOKEN_NULL &&
                          (tokens.empty() || tokens.back() != eos);

  // Reserve space for EOS token if needed
  int effective_limit = append_eos ? limit_size - 1 : limit_size;

  if ((int)tokens.size() > effective_limit) {
    LLAMA_LOG_WARN("Prompt too long %ld, limit size %d, truncating...",
                   tokens.size(), limit_size);
    new_tokens.resize(effective_limit);
  }

  if (append_eos && !new_tokens.empty() && new_tokens.back() != eos) {
    new_tokens.push_back(eos);
  }

  return new_tokens;
}

void EmbeddingRequestHandler::handle(const std::string &input_prompt,
                                     ServerSlot *slot) {
  // BERT-style embedding vocabs have no EOS token; let their tokenizer add
  // the special tokens they expect ([CLS] ... [SEP]) instead of appending EOS
  const bool has_eos = this->llama_->get_token_eos() != LLAMA_TOKEN_NULL;
  auto tokens = common_tokenize(this->llama_->get_ctx(), input_prompt,
                                /*add_special=*/!has_eos, true);
  LLAMA_LOG_INFO("Tokenized prompt to %ld tokens", tokens.size());
  tokens = this->truncate_tokens(tokens, llama_n_batch(this->llama_->get_ctx()),
                                 true);

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->sampler = common_sampler_init(this->llama_->get_model(),
                                      this->llama_->params.sampling);
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

  // Vocabs without an EOS token (BERT-style rerankers) must never get
  // LLAMA_TOKEN_NULL appended: it would be mistaken for a media placeholder
  const llama_token eos = this->llama_->get_token_eos();
  const bool append_eos = add_eos && eos != LLAMA_TOKEN_NULL &&
                          (tokens.empty() || tokens.back() != eos);

  // Reserve space for EOS token if needed
  int effective_limit = append_eos ? limit_size - 1 : limit_size;

  if ((int)tokens.size() > effective_limit) {
    LLAMA_LOG_WARN("Prompt too long %ld, limit size %d, truncating...",
                   tokens.size(), limit_size);
    new_tokens.resize(effective_limit);
  }

  if (append_eos && !new_tokens.empty() && new_tokens.back() != eos) {
    new_tokens.push_back(eos);
  }

  return new_tokens;
}

void RerankRequestHandler::handle(const std::string &query,
                                  const std::string &document,
                                  ServerSlot *slot) {
  // BERT-style vocabs have no EOS token; never push special tokens the vocab
  // does not define so no LLAMA_TOKEN_NULL ever reaches the prompt (e.g. the
  // query/document separator is [SEP] for BERT rerankers)
  const llama_token bos = this->llama_->get_token_bos();
  const llama_token eos = this->llama_->get_token_eos();
  const llama_token sep = this->llama_->get_token_sep();

  std::vector<llama_token> tokens;
  if (bos != LLAMA_TOKEN_NULL) {
    tokens.push_back(bos);
  }

  auto tokens_query =
      common_tokenize(this->llama_->get_vocab(), query, false, true);
  auto truncated_query = this->truncate_tokens(
      tokens_query, (int)(this->llama_->params.n_batch / 2) - 2, true);
  tokens.insert(tokens.end(), truncated_query.begin(), truncated_query.end());

  if (eos != LLAMA_TOKEN_NULL) {
    tokens.push_back(eos);
  }

  if (sep != LLAMA_TOKEN_NULL) {
    tokens.push_back(sep);
  }

  auto tokens_document =
      common_tokenize(this->llama_->get_vocab(), document, false, true);
  auto truncated_document = this->truncate_tokens(
      tokens_document, (int)(this->llama_->params.n_batch / 2) - 2, true);
  tokens.insert(tokens.end(), truncated_document.begin(),
                truncated_document.end());

  if (eos != LLAMA_TOKEN_NULL) {
    tokens.push_back(eos);
  }

  tokens = this->truncate_tokens(tokens, llama_n_batch(this->llama_->get_ctx()),
                                 true);

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->sampler = common_sampler_init(this->llama_->get_model(),
                                      this->llama_->params.sampling);
  slot->prompt_tokens = tokens;
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = SERVER_TASK_TYPE_RERANK;
  slot->state = SLOT_STATE_STARTED;
}

// DecisionRequestHandler implementation
void DecisionRequestHandler::handle(const std::vector<llama_token> &tokens,
                                    const DecisionTaskMeta &meta,
                                    ServerSlot *slot) {
  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->sampler = common_sampler_init(this->llama_->get_model(),
                                      this->llama_->params.sampling);
  slot->prompt_tokens = tokens;
  slot->decision = meta;
  slot->cache_prompt = false;
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = SERVER_TASK_TYPE_DECISION;
  slot->state = SLOT_STATE_STARTED;
}

void DecisionRequestHandler::handle_prefilled(const DecisionTaskMeta &meta,
                                              ServerSlot *slot) {
  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->sampler = common_sampler_init(this->llama_->get_model(),
                                      this->llama_->params.sampling);
  slot->decision = meta;
  slot->cache_prompt = false;
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = SERVER_TASK_TYPE_DECISION;
  slot->state = SLOT_STATE_STARTED;
}

// CompletionRequestHandler implementation
void CompletionRequestHandler::handle(
    const std::string &input_prompt, ServerSlot *slot,
    common_params_sampling sparams,
    std::function<void(CompletionOutput, ServerSlot *)> callback,
    std::vector<std::string> stop, bool /*reset*/) {

  slot->prompt_tokens.clear();
  std::string full_prompt;

  if (this->llama_->params.input_prefix.size() > 0) {
    full_prompt += this->llama_->params.input_prefix;
  }
  full_prompt += input_prompt;
  if (this->llama_->params.input_suffix.size() > 0) {
    full_prompt += this->llama_->params.input_suffix;
  }

  slot->prompt_tokens =
      common_tokenize(this->llama_->get_ctx(), full_prompt, false, true);
  LLAMA_LOG_INFO("Full prompt: '%s'", full_prompt.c_str());
  LLAMA_LOG_INFO("Tokenized prompt to %ld tokens", slot->prompt_tokens.size());

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->params.sampling = sparams;
  slot->sampler =
      common_sampler_init(this->llama_->get_model(), slot->params.sampling);
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

  slot->chat_session = std::move(chat_context.chat_session);
  slot->params.sampling = chat_context.sparams;

  if (slot->sampler != nullptr) {
    common_sampler_free(slot->sampler);
  }

  slot->prompt_tokens = common_tokenize(
      this->llama_->get_ctx(), slot->chat_session.prompt().str(), false, true);
  slot->sampler =
      common_sampler_init(this->llama_->get_model(), slot->params.sampling);
  slot->stream_callback = callback;
  slot->chat_format = slot->chat_session.format();
  LLAMA_LOG_INFO("Prompt tokens size: %ld", slot->prompt_tokens.size());
  slot->task_type = SERVER_TASK_TYPE_COMPLETION;
  slot->state = SLOT_STATE_STARTED;
}
