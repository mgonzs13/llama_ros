// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
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

#ifndef LLAMA_ROS__SERVER_TASK_RESULT_HPP
#define LLAMA_ROS__SERVER_TASK_RESULT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"

#include "llama_ros/types.hpp"

namespace llama_ros {

/**
 * @brief Base class for server task results.
 *
 * This is the polymorphic base for all task result types,
 * including embeddings, reranking, and completion results.
 */
struct ServerTaskResult {
  /// @brief The unique identifier for the task.
  uint64_t id;

  /// @brief The slot ID that processed this task.
  int id_slot;

  /// @brief Virtual destructor for proper polymorphic cleanup.
  virtual ~ServerTaskResult() = default;
};

/**
 * @brief Result of an embedding generation task.
 */
struct ServerTaskResultEmbedding : ServerTaskResult {
  /// @brief The generated embeddings (one vector per token or per sequence).
  std::vector<std::vector<float>> embeddings;

  /// @brief The number of tokens processed.
  int32_t n_tokens;
};

/**
 * @brief Result of a reranking task.
 */
struct ServerTaskResultRerank : ServerTaskResult {
  /// @brief The relevance score for the document.
  float score = -1e6;
};

/**
 * @brief Partial result of a completion task (used during streaming).
 */
struct ServerTaskResultCompletionPartial : ServerTaskResult {
  /// @brief The partial content generated so far.
  std::string content;

  /// @brief The tokens generated so far.
  llama_tokens tokens;

  /// @brief The number of tokens decoded.
  int32_t n_decoded;

  /// @brief The number of prompt tokens.
  int32_t n_prompt_tokens;

  /// @brief The probability output for the current token.
  TokenProb prob_output;

  /// @brief Build information for debugging.
  std::string build_info;

  /// @brief The stopping condition.
  StopType stop;

  /// @brief Whether post-sampling probabilities are included.
  bool post_sampling_probs;

  /// @brief The OpenAI-compatible model name.
  std::string oaicompat_model;

  /// @brief The OpenAI-compatible completion ID.
  std::string oaicompat_cmpl_id;

  /// @brief The OpenAI-compatible message diffs for streaming.
  std::vector<common_chat_msg_diff> oaicompat_msg_diffs;
};

/**
 * @brief Full result of a completion task.
 */
struct ServerTaskResultCompletion : ServerTaskResult {
  /// @brief The content of the response.
  std::string content;

  /// @brief The list of token IDs in the response.
  std::vector<int> tokens;

  /// @brief Whether the response was streamed.
  bool stream;

  /// @brief The prompt used to generate the response.
  std::string prompt;

  /// @brief Build information for debugging.
  std::string build_info;

  /// @brief The number of tokens decoded in the response.
  int32_t n_decoded;

  /// @brief The number of tokens in the prompt.
  int32_t n_prompt_tokens;

  /// @brief The stop condition for the response generation.
  StopType stop;

  /// @brief Whether post-sampling probabilities are included.
  bool post_sampling_probs;

  /// @brief The output probabilities for selected tokens.
  std::vector<SelectedLogProb> probs_output;

  /// @brief Additional fields included in the response.
  std::vector<std::string> response_fields;

  /// @brief The OpenAI-compatible chat format.
  common_chat_format oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

  /// @brief The OpenAI-compatible model name.
  std::string oaicompat_model;

  /// @brief The OpenAI-compatible completion ID.
  std::string oaicompat_cmpl_id;

  /// @brief The OpenAI-compatible chat message.
  common_chat_msg oaicompat_msg;

  /// @brief The OpenAI-compatible message diffs for streaming.
  std::vector<common_chat_msg_diff> oaicompat_msg_diffs;
};

/// @brief Unique pointer type alias for ServerTaskResult.
using ServerTaskResultPtr = std::unique_ptr<ServerTaskResult>;

} // namespace llama_ros

#endif // LLAMA_ROS__SERVER_TASK_RESULT_HPP
