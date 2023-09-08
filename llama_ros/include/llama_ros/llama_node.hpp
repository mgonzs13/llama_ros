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

#ifndef LLAMA_ROS__LLAMA_NODE_HPP
#define LLAMA_ROS__LLAMA_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <memory>
#include <string>

#include "common.h"
#include "llama.h"
#include "llama_msgs/action/generate_response.hpp"
#include "llama_msgs/srv/generate_embeddings.hpp"
#include "llama_msgs/srv/tokenize.hpp"
#include "llama_ros/llama.hpp"
namespace llama_ros {

class LlamaNode : public rclcpp::Node {

  using GenerateResponse = llama_msgs::action::GenerateResponse;
  using GoalHandleGenerateResponse =
      rclcpp_action::ServerGoalHandle<GenerateResponse>;

public:
  LlamaNode();

private:
  std::shared_ptr<Llama> llama;

  // ros2
  rclcpp::Service<llama_msgs::srv::Tokenize>::SharedPtr tokenize_service_;
  rclcpp::Service<llama_msgs::srv::GenerateEmbeddings>::SharedPtr
      generate_embeddings_service_;
  rclcpp_action::Server<GenerateResponse>::SharedPtr
      generate_response_action_server_;
  GenerateResponse::Goal current_goal_;
  std::shared_ptr<GoalHandleGenerateResponse> goal_handle_;
  std::mutex handle_accepted_mtx_;

  // methods
  void load_params(gpt_params &params);
  void tokenize_service_callback(
      const std::shared_ptr<llama_msgs::srv::Tokenize::Request> request,
      std::shared_ptr<llama_msgs::srv::Tokenize::Response> response);
  void generate_embeddings_service_callback(
      const std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Request>
          request,
      std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Response> response);

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const GenerateResponse::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleGenerateResponse> goal_handle);
  void handle_accepted(
      const std::shared_ptr<GoalHandleGenerateResponse> goal_handle);

  void execute(const std::shared_ptr<GoalHandleGenerateResponse> goal_handle);
  void send_text(const completion_output &completion);
};

} // namespace llama_ros

#endif
