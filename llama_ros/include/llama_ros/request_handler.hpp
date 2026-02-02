// MIT License
//
// Copyright (c) 2023 Miguel Ángel González Santamarta
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

#ifndef LLAMA_ROS__REQUEST_HANDLER_HPP
#define LLAMA_ROS__REQUEST_HANDLER_HPP

#include <functional>
#include <string>
#include <vector>

#include "common.h"
#include "llama.h"

namespace llama_utils {
  struct ChatCompletionsContext;
}

namespace llama_ros {

// Forward declarations
class ServerSlot;
class Llama;
struct CompletionOutput;

/**
 * @brief Base interface for request handlers.
 *
 * This provides a common interface for different types of request handlers
 * (embeddings, reranking, completion, etc.).
 */
class RequestHandler {
public:
  /**
   * @brief Constructor for RequestHandler.
   *
   * @param llama Pointer to the Llama instance that owns this handler.
   */
  explicit RequestHandler(Llama *llama) : llama_(llama) {}

  /**
   * @brief Virtual destructor.
   */
  virtual ~RequestHandler() = default;

protected:
  Llama *llama_;
  
  // Declare friendship with Llama to access protected members
  friend class Llama;
};

/**
 * @brief Handles embedding generation requests.
 */
class EmbeddingRequestHandler : public RequestHandler {
public:
  explicit EmbeddingRequestHandler(Llama *llama) : RequestHandler(llama) {}

  /**
   * @brief Prepares a slot for embedding generation.
   *
   * @param input_prompt The input text to generate embeddings for.
   * @param slot The slot to prepare.
   */
  void handle(const std::string &input_prompt, ServerSlot *slot);

private:
  std::vector<llama_token> truncate_tokens(const std::vector<llama_token> &tokens, 
                                          int limit_size, bool add_eos = true);
};

/**
 * @brief Handles reranking requests.
 */
class RerankRequestHandler : public RequestHandler {
public:
  explicit RerankRequestHandler(Llama *llama) : RequestHandler(llama) {}

  /**
   * @brief Prepares a slot for reranking.
   *
   * @param query The query string.
   * @param document The document string to rank.
   * @param slot The slot to prepare.
   */
  void handle(const std::string &query, const std::string &document, ServerSlot *slot);

private:
  std::vector<llama_token> truncate_tokens(const std::vector<llama_token> &tokens, 
                                          int limit_size, bool add_eos = true);
};

/**
 * @brief Handles text completion requests.
 */
class CompletionRequestHandler : public RequestHandler {
public:
  explicit CompletionRequestHandler(Llama *llama) : RequestHandler(llama) {}

  /**
   * @brief Prepares a slot for text completion.
   *
   * @param input_prompt The input prompt.
   * @param slot The slot to prepare.
   * @param sparams Sampling parameters.
   * @param callback Callback for streaming results.
   * @param stop Stop sequences.
   * @param reset Whether to reset the slot.
   */
  void handle(const std::string &input_prompt, 
             ServerSlot *slot,
             struct common_params_sampling sparams,
             std::function<void(struct CompletionOutput, ServerSlot *)> callback,
             std::vector<std::string> stop, 
             bool reset);
};

/**
 * @brief Handles chat completion requests.
 */
class ChatCompletionRequestHandler : public RequestHandler {
public:
  explicit ChatCompletionRequestHandler(Llama *llama) : RequestHandler(llama) {}

  /**
   * @brief Prepares a slot for chat completion.
   *
   * @param chat_context The chat context containing messages and parameters.
   * @param slot The slot to prepare.
   * @param callback Callback for streaming results.
   */
  void handle(llama_utils::ChatCompletionsContext chat_context, 
             ServerSlot *slot,
             std::function<void(struct CompletionOutput, ServerSlot *)> callback);
};

} // namespace llama_ros

#endif // LLAMA_ROS__REQUEST_HANDLER_HPP
