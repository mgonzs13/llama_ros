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

#ifndef LLAMA_ROS__SERVER_SLOT_HPP
#define LLAMA_ROS__SERVER_SLOT_HPP

#include <cstdint>
#include <functional>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "sampling.h"
#include <mtmd.h>

#include "llama_ros/types.hpp"

using json = nlohmann::ordered_json;

namespace llama_ros {

/**
 * @brief Represents a processing slot in the server.
 *
 * Each slot can handle one request at a time and maintains its own
 * state, sampling context, and generation history. Multiple slots
 * allow concurrent request processing.
 */
class ServerSlot {
public:
  /**
   * @brief A callback function type for handling generated responses.
   */
  using GenerateResponseCallback =
      std::function<void(CompletionOutput, ServerSlot *)>;

  /// @brief The unique identifier for the slot.
  int id;

  /// @brief The unique identifier for the goal associated with the slot.
  uint64_t goal_id;

  /// @brief The batch associated with the slot.
  llama_batch batch;

  /// @brief The context associated with the slot.
  llama_context *ctx;

  /// @brief The sampler used for generating responses in the slot.
  common_sampler *sampler;

  /// @brief The LoRA adapters used in the slot.
  std::vector<common_adapter_lora_info> lora_adapters;

  /// @brief The task type associated with the slot.
  ServerTaskType task_type = SERVER_TASK_TYPE_COMPLETION;

  /// @brief The token sampled in the slot.
  llama_token sampled;

  /// @brief The current state of the slot.
  SlotState state = SLOT_STATE_IDLE;

  /// @brief The JSON schema for structured output.
  json json_schema;

  /// @brief The word that triggered the stop.
  std::string stopping_word;

  /// @brief Whether the slot has a next token to generate.
  bool has_next_token = true;

  /// @brief Whether the generated text contains a new line.
  bool has_new_line = false;

  /// @brief The number of past tokens processed.
  int32_t n_past = 0;

  /// @brief The context size for this slot.
  int32_t n_ctx = 0;

  /// @brief The number of tokens consumed.
  int32_t n_consumed = 0;

  /// @brief The maximum number of tokens to predict.
  int32_t n_predict = -1;

  /// @brief The batch index for this slot.
  int32_t i_batch = -1;

  /// @brief The number of tokens decoded.
  int32_t n_decoded = 0;

  /// @brief The group attention index.
  int32_t ga_i = 0;

  /// @brief The number of text characters sent.
  size_t n_sent_text = 0;

  /// @brief Whether to stream the response.
  bool stream;

  /// @brief The callback for streaming generated tokens.
  GenerateResponseCallback stream_callback = nullptr;

  /**
   * @brief Parameters for slot configuration.
   */
  struct slot_params {
    /// @brief Number of tokens to keep from the initial prompt.
    int32_t n_keep = 0;

    /// @brief Number of tokens to discard during context shift.
    int32_t n_discard = 0;

    /// @brief Maximum number of tokens to predict.
    int32_t n_predict = -1;

    /// @brief Indentation level for stopping.
    int32_t n_indent = 0;

    /// @brief LoRA adapters for this slot.
    std::vector<common_adapter_lora_info> lora;

    /// @brief Stop sequences (antiprompts).
    std::vector<std::string> antiprompt;

    /// @brief Sampling parameters.
    common_params_sampling sampling;

    /// @brief Whether to enable verbose output.
    bool verbose = false;

    /// @brief The OpenAI-compatible model name.
    std::string oaicompat_model;

    /// @brief The OpenAI-compatible completion ID.
    std::string oaicompat_cmpl_id;

    /// @brief The OpenAI-compatible chat syntax parameters.
    common_chat_parser_params oaicompat_chat_syntax;
  } params;

  /// @brief The prompt tokens.
  std::vector<llama_token> prompt_tokens;

  /// @brief The total number of prompt tokens.
  int32_t n_prompt_tokens = 0;

  /// @brief The number of prompt tokens processed so far.
  int32_t n_prompt_tokens_processed = 0;

  /// @brief The position of the last newline in generated text.
  size_t last_nl_pos = 0;

  /// @brief The parsed chat message.
  common_chat_msg chat_msg;

  /// @brief The chat format being used.
  common_chat_format chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

  /// @brief Generated tool call IDs.
  std::vector<std::string> generated_tool_call_ids;

  /// @brief The current stop condition.
  StopType stop;

  /// @brief The full generated text.
  std::string generated_text;

  /// @brief The generated token IDs.
  llama_tokens generated_tokens;

  /// @brief The generated token probabilities.
  std::vector<std::vector<TokenProb>> generated_probs;

  /// @brief Previous performance statistics.
  llama_perf_context_data prev_stat_usage;

  /// @brief Map from position to media chunk for multimodal processing.
  std::unordered_map<llama_pos, mtmd::input_chunk_ptr> map_pos_to_media;

  /**
   * @brief Resets the slot to its initial state.
   */
  void reset();

  /**
   * @brief Updates the chat message and computes diffs.
   *
   * @param diffs Output vector of message diffs.
   * @return Reference to the updated chat message.
   */
  const common_chat_msg &
  update_chat_msg(std::vector<common_chat_msg_diff> &diffs);

  /**
   * @brief Releases the slot, making it available for reuse.
   */
  void release();

  /**
   * @brief Checks if the slot is currently processing a request.
   *
   * @return True if the slot is processing, false otherwise.
   */
  inline bool is_processing() const { return this->state != SLOT_STATE_IDLE; }

  /**
   * @brief Searches for stopping strings in the generated text.
   *
   * @param text The text to search in.
   * @param last_token_size The size of the last token.
   * @param is_full_stop Whether to search for full stops.
   * @return The position of the stopping string, or npos if not found.
   */
  size_t find_stopping_strings(const std::string &text,
                               const size_t last_token_size, bool is_full_stop);
};

} // namespace llama_ros

#endif // LLAMA_ROS__SERVER_SLOT_HPP
