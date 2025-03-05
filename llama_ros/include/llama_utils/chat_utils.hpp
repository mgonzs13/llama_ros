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

#ifndef LLAMA_UTILS__CHAT_UTILS_HPP
#define LLAMA_UTILS__CHAT_UTILS_HPP

#include <llama_msgs/action/generate_chat_completions.hpp>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "chat.h"
#include "common.h"
#include "llama.h"
#include "llama_ros/llama.hpp"

namespace llama_utils {
struct LogProb {
  std::string token;
  float probability;
  std::string text;
};

struct SelectedLogProb {
  LogProb chosen_token;
  std::vector<LogProb> data;
};

struct ResponseResult {
  int index;
  std::string content;
  std::vector<int> tokens;

  bool stream;
  std::string prompt;
  std::string build_info;

  int32_t n_decoded;
  int32_t n_prompt_tokens;
  int32_t n_tokens_cached;
  llama_ros::StopType stop;

  bool post_sampling_probs;
  std::vector<SelectedLogProb> probs_output;
  std::vector<std::string> response_fields;

  common_chat_format oaicompat_chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
  std::string oaicompat_model;
  std::string oaicompat_cmpl_id;
};

common_chat_templates_inputs parse_chat_completions_goal(
    const std::shared_ptr<
        const llama_msgs::action::GenerateChatCompletions::Goal>
        goal);

llama_msgs::action::GenerateChatCompletions::Result
generate_chat_completions_result(const ResponseResult &result);
} // namespace llama_utils
#endif
