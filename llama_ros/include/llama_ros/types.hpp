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

#ifndef LLAMA_ROS__TYPES_HPP
#define LLAMA_ROS__TYPES_HPP

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"

namespace llama_ros {

/**
 * @brief Options for the OAI-compatible response parser.
 */
struct OAICompactParserOptions {
  /// @brief Whether to use Jinja templates.
  bool use_jinja;

  /// @brief Whether to prefill the assistant response.
  bool prefill_assistant;

  /// @brief The reasoning format for the response.
  common_reasoning_format reasoning_format;

  /// @brief Additional keyword arguments for the chat template.
  std::map<std::string, std::string> chat_template_kwargs;

  /// @brief Pointer to the chat templates.
  common_chat_templates *tmpls;

  /// @brief Whether to allow image inputs.
  bool allow_image;

  /// @brief Whether to allow audio inputs.
  bool allow_audio;

  /// @brief Whether to enable thinking/reasoning output.
  bool enable_thinking = true;
};

/**
 * @brief Represents the probability of a token.
 */
struct TokenProb {
  /// @brief The token ID.
  llama_token token;

  /// @brief The probability of the token.
  float probability;
};

/**
 * @brief Represents a Low-Rank Adaptation (LoRA) configuration.
 */
struct LoRA {
  /// @brief The ID of the LoRA configuration.
  int id;

  /// @brief The file path to the LoRA model.
  std::string path;

  /// @brief The scaling factor for the LoRA model.
  float scale;
};

/**
 * @brief Represents the output of a completion operation.
 */
struct CompletionOutput {
  /// @brief The probabilities of tokens in the completion.
  std::vector<TokenProb> probs;

  /// @brief The token generated in the completion.
  llama_token token;

  /// @brief The text to send for this token.
  std::string text_to_send;
};

/**
 * @brief Represents the stopping condition for a process.
 */
enum StopType {
  NO_STOP,      ///< No stopping condition.
  FULL_STOP,    ///< Full stop condition.
  PARTIAL_STOP, ///< Partial stop condition.
  CANCEL,       ///< Cancel the process.
  ABORT         ///< Abort the process.
};

/**
 * @brief Represents the output of a response generation process.
 */
struct ResponseOutput {
  /// @brief The list of completion outputs.
  std::vector<CompletionOutput> completions;

  /// @brief The stopping condition for the response generation.
  StopType stop;
};

/**
 * @brief Represents the output of an embedding generation process.
 */
struct EmbeddingsOutput {
  /// @brief The generated embeddings.
  std::vector<float> embeddings;

  /// @brief The number of tokens used to generate the embeddings.
  int32_t n_tokens;
};

/**
 * @brief Represents the state of a server slot.
 */
enum SlotState {
  SLOT_STATE_IDLE,              ///< The slot is idle.
  SLOT_STATE_STARTED,           ///< The slot has started processing.
  SLOT_STATE_PROCESSING_PROMPT, ///< The slot is processing the prompt.
  SLOT_STATE_DONE_PROMPT, ///< The slot has finished processing the prompt.
  SLOT_STATE_GENERATING   ///< The slot is generating tokens.
};

/**
 * @brief Represents the type of server task.
 */
enum ServerTaskType {
  SERVER_TASK_TYPE_COMPLETION, ///< Text completion task.
  SERVER_TASK_TYPE_EMBEDDING,  ///< Embedding generation task.
  SERVER_TASK_TYPE_RERANK,     ///< Document reranking task.
  SERVER_TASK_TYPE_CANCEL,     ///< Task cancellation.
};

/**
 * @brief Represents a log probability for a token.
 */
struct LogProb {
  /// @brief The token ID.
  int token;

  /// @brief The log probability of the token.
  float probability;

  /// @brief The text representation of the token.
  std::string text;
};

/**
 * @brief Represents a selected log probability and its associated data.
 */
struct SelectedLogProb {
  /// @brief The chosen token and its log probability.
  LogProb chosen_token;

  /// @brief A list of log probabilities for other tokens.
  std::vector<LogProb> data;
};

} // namespace llama_ros

#endif // LLAMA_ROS__TYPES_HPP
