// MIT License

// Copyright (c) 2023  Miguel Ángel González Santamarta

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef LLAMA_NODE_HPP
#define LLAMA_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <memory>
#include <string>

#include "llama.h"
#include "llama_msgs/action/gpt.hpp"
#include "llama_ros/llama.hpp"

namespace llama_ros {

class LlamaNode : public rclcpp::Node {

  using GPT = llama_msgs::action::GPT;
  using GoalHandleGPT = rclcpp_action::ServerGoalHandle<GPT>;

public:
  LlamaNode();

private:
  std::shared_ptr<Llama> llama;

  // ros2
  rclcpp_action::Server<GPT>::SharedPtr gpt_action_server_;
  GPT::Goal current_goal_;
  std::shared_ptr<GoalHandleGPT> goal_handle_;
  std::mutex handle_accepted_mtx_;

  // methods
  void process_initial_prompt(std::string prompt);

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const GPT::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleGPT> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleGPT> goal_handle);

  void execute(const std::shared_ptr<GoalHandleGPT> goal_handle);
  void send_text(const std::string &text);
};

} // namespace llama_ros

#endif
