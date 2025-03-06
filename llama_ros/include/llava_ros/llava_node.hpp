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

#ifndef LLAVA_ROS__LLAVA_NODE_HPP
#define LLAVA_ROS__LLAVA_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <memory>
#include <string>

#include "common.h"

#include "llama_msgs/action/generate_response.hpp"
#include "llama_msgs/action/generate_chat_completions.hpp"
#include "llama_ros/llama_node.hpp"
#include "llava_ros/llava.hpp"

namespace llava_ros {

class LlavaNode : public llama_ros::LlamaNode {

  using GenerateResponse = llama_msgs::action::GenerateResponse;
  using GoalHandleGenerateResponse =
      rclcpp_action::ServerGoalHandle<GenerateResponse>;
  using GenerateChatCompletions = llama_msgs::action::GenerateChatCompletions;
  using GoalHandleGenerateChatCompletions =
      rclcpp_action::ServerGoalHandle<GenerateChatCompletions>;
    

public:
  LlavaNode();

  std::string base64_encode(unsigned char const *bytes_to_encode, size_t in_len,
                            bool url = false);

protected:
  void create_llama() override;
  bool goal_empty(std::shared_ptr<const GenerateResponse::Goal> goal) override;
  void execute(
      const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) override;

  bool goal_empty_chat_completions(
      std::shared_ptr<const GenerateChatCompletions::Goal> goal) override;
  void execute_chat_completions(
      const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle)
      override;
};

} // namespace llava_ros

#endif
