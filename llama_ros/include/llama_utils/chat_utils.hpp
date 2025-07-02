// MIT License
//
// Copyright (c) 2025 Miguel Ángel González Santamarta
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

#ifndef LLAMA_UTILS__CHAT_UTILS_HPP
#define LLAMA_UTILS__CHAT_UTILS_HPP

#include <common.h>
#include <llama_msgs/action/generate_chat_completions.hpp>
#include <memory>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "chat.h"
#include "llama_msgs/msg/chat_reasoning_format.hpp"
#include "llama_ros/llama.hpp"
#include "llama_utils/llama_params.hpp"

namespace llama_utils {

/**
 * @brief Represents a log probability for a token.
 */
struct LogProb {
  /**
   * @brief The token ID.
   */
  int token;

  /**
   * @brief The log probability of the token.
   */
  float probability;

  /**
   * @brief The text representation of the token.
   */
  std::string text;
};

/**
 * @brief Represents a selected log probability and its associated data.
 */
struct SelectedLogProb {
  /**
   * @brief The chosen token and its log probability.
   */
  LogProb chosen_token;

  /**
   * @brief A list of log probabilities for other tokens.
   */
  std::vector<LogProb> data;
};

/**
 * @brief Represents the result of a chat response.
 */
struct ResponseResult {
  /**
   * @brief The index of the response in a sequence.
   */
  int index;

  /**
   * @brief The content of the chat response.
   */
  std::string content;

  /**
   * @brief The list of token IDs in the response.
   */
  std::vector<int> tokens;

  /**
   * @brief Indicates if the response is streamed.
   */
  bool stream;

  /**
   * @brief The prompt used to generate the response.
   */
  std::string prompt;

  /**
   * @brief Build information for debugging purposes.
   */
  std::string build_info;

  /**
   * @brief The number of tokens decoded in the response.
   */
  int32_t n_decoded;

  /**
   * @brief The number of tokens in the prompt.
   */
  int32_t n_prompt_tokens;

  /**
   * @brief The number of tokens retrieved from the cache.
   */
  int32_t n_tokens_cached;

  /**
   * @brief The stop condition for the response generation.
   */
  llama_ros::StopType stop;

  /**
   * @brief Indicates if post-sampling probabilities are included.
   */
  bool post_sampling_probs;

  /**
   * @brief The output probabilities for selected tokens.
   */
  std::vector<SelectedLogProb> probs_output;

  /**
   * @brief Additional fields included in the response.
   */
  std::vector<std::string> response_fields;

  /**
   * @brief The OpenAI-compatible chat format.
   */
  common_chat_format oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;

  /**
   * @brief The OpenAI-compatible model name.
   */
  std::string oaicompat_model;

  /**
   * @brief The OpenAI-compatible completion ID.
   */
  std::string oaicompat_cmpl_id;

  /**
   * @brief The OpenAI-compatible chat syntax. Used while streaming the
   * response.
   */
  common_chat_msg chat_msg;
};

/**
 * @brief Generates a random alphanumeric string.
 *
 * @param string_size The size of the string to generate. Default is 32.
 * @return A random alphanumeric string.
 */
static inline std::string random_string(int string_size) {
  static const std::string str(
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");

  std::random_device rd;
  std::mt19937 generator(rd());

  std::string result(string_size, ' ');

  for (int i = 0; i < string_size; ++i) {
    result[i] = str[generator() % str.size()];
  }

  return result;
}

static inline std::string random_string() { return random_string(32); }

/**
 * @brief Generates a unique chat completion ID.
 *
 * @return A unique chat completion ID.
 */
inline std::string gen_chatcmplid() { return "chatcmpl-" + random_string(); }

/**
 * @brief Computes the logit (logarithm of odds) of a value.
 *
 * @param x The input value.
 * @return The logit of the input value.
 */
inline float logit(float x) {
  return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

/**
 * @brief Parses a chat tool choice from an integer.
 *
 * @param choice The integer representing the tool choice.
 * @return The parsed chat tool choice.
 */
common_chat_tool_choice parse_chat_tool_choice(int choice);

common_reasoning_format parse_reasoning_format(const int reasoning_format);

/**
 * @brief Parses the goal for generating chat completions.
 *
 * @param goal The goal shared pointer from the action server.
 * @return The parsed chat templates inputs.
 */
struct common_chat_templates_inputs parse_chat_completions_goal(
    const std::shared_ptr<
        const llama_msgs::action::GenerateChatCompletions::Goal>
        goal);

/**
 * @brief Generates the result for a chat completion action.
 *
 * @param result The response result to convert.
 * @return The generated result for the action.
 */
llama_msgs::action::GenerateChatCompletions::Result
generate_chat_completions_result(const ResponseResult &result);

/**
 * @brief Generates feedback for a chat completion action.
 *
 * @param result The response result to convert.
 * @return A vector of feedback messages for the action.
 */
std::vector<llama_msgs::action::GenerateChatCompletions::Feedback>
generate_chat_completions_feedback(
    const ResponseResult &result,
    std::vector<common_chat_msg_diff> deltas = {});

/**
 * @brief Represents the context for chat completions.
 */
struct ChatCompletionsContext {
  common_chat_syntax oaicompat_chat_syntax;
  common_params_sampling sparams;
  common_chat_templates_inputs prompt_format_config;
  common_chat_params chat_prompt_instance;
};

/**
 * @brief Prepares the context for chat completions.
 *
 * @param goal The goal shared pointer from the action server.
 * @param llama The Llama instance.
 * @return The prepared chat completions context.
 */
ChatCompletionsContext prepare_chat_completions_call(
    const std::shared_ptr<
        const llama_msgs::action::GenerateChatCompletions::Goal> &goal,
    llama_ros::Llama *llama);

} // namespace llama_utils

#endif
