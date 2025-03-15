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

#ifndef LLAMA_ROS__LLAMA_NODE_HPP
#define LLAMA_ROS__LLAMA_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include <memory>
#include <string>

#include "common.h"
#include "llama.h"

#include "llama_msgs/action/generate_chat_completions.hpp"
#include "llama_msgs/action/generate_response.hpp"
#include "llama_msgs/srv/detokenize.hpp"
#include "llama_msgs/srv/generate_embeddings.hpp"
#include "llama_msgs/srv/get_metadata.hpp"
#include "llama_msgs/srv/list_lo_r_as.hpp"
#include "llama_msgs/srv/rerank_documents.hpp"
#include "llama_msgs/srv/tokenize.hpp"
#include "llama_msgs/srv/update_lo_r_as.hpp"
#include "llama_ros/llama.hpp"
#include "llama_utils/llama_params.hpp"

namespace llama_ros {

using CallbackReturn =
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class LlamaNode : public rclcpp_lifecycle::LifecycleNode {

  using GenerateResponse = llama_msgs::action::GenerateResponse;
  using GoalHandleGenerateResponse =
      rclcpp_action::ServerGoalHandle<GenerateResponse>;
  using GenerateChatCompletions = llama_msgs::action::GenerateChatCompletions;
  using GoalHandleGenerateChatCompletions =
      rclcpp_action::ServerGoalHandle<GenerateChatCompletions>;

public:
  LlamaNode();

  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &);
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &);
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &);
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &);
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &);

protected:
  std::unique_ptr<Llama> llama;
  bool params_declared;
  struct llama_utils::LlamaParams params;
  std::shared_ptr<GoalHandleGenerateResponse> goal_handle_;
  std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle_chat_;

  virtual void create_llama();
  void destroy_llama();

  virtual bool goal_empty(std::shared_ptr<const GenerateResponse::Goal> goal);
  virtual void
  execute(const std::shared_ptr<GoalHandleGenerateResponse> goal_handle);
  void send_text(const struct CompletionOutput &completion);

  virtual bool goal_empty_chat_completions(
      std::shared_ptr<const GenerateChatCompletions::Goal> goal);
  virtual void execute_chat_completions(
      const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle);
  void send_text_chat_completions(const struct CompletionOutput &completion);

private:
  // ros2
  rclcpp::Service<llama_msgs::srv::GetMetadata>::SharedPtr
      get_metadata_service_;
  rclcpp::Service<llama_msgs::srv::Tokenize>::SharedPtr tokenize_service_;
  rclcpp::Service<llama_msgs::srv::Detokenize>::SharedPtr detokenize_service_;
  rclcpp::Service<llama_msgs::srv::GenerateEmbeddings>::SharedPtr
      generate_embeddings_service_;
  rclcpp::Service<llama_msgs::srv::RerankDocuments>::SharedPtr
      rerank_documents_service_;
  rclcpp::Service<llama_msgs::srv::ListLoRAs>::SharedPtr list_loras_service_;
  rclcpp::Service<llama_msgs::srv::UpdateLoRAs>::SharedPtr
      update_loras_service_;
  rclcpp_action::Server<GenerateResponse>::SharedPtr
      generate_response_action_server_;
  rclcpp_action::Server<GenerateChatCompletions>::SharedPtr
      generate_chat_completions_action_server_;

  // methods
  void get_metadata_service_callback(
      const std::shared_ptr<llama_msgs::srv::GetMetadata::Request> request,
      std::shared_ptr<llama_msgs::srv::GetMetadata::Response> response);

  void tokenize_service_callback(
      const std::shared_ptr<llama_msgs::srv::Tokenize::Request> request,
      std::shared_ptr<llama_msgs::srv::Tokenize::Response> response);
  void detokenize_service_callback(
      const std::shared_ptr<llama_msgs::srv::Detokenize::Request> request,
      std::shared_ptr<llama_msgs::srv::Detokenize::Response> response);

  void generate_embeddings_service_callback(
      const std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Request>
          request,
      std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Response> response);
  void rerank_documents_service_callback(
      const std::shared_ptr<llama_msgs::srv::RerankDocuments::Request> request,
      std::shared_ptr<llama_msgs::srv::RerankDocuments::Response> response);

  void list_loras_service_callback(
      const std::shared_ptr<llama_msgs::srv::ListLoRAs::Request> request,
      std::shared_ptr<llama_msgs::srv::ListLoRAs::Response> response);
  void update_loras_service_callback(
      const std::shared_ptr<llama_msgs::srv::UpdateLoRAs::Request> request,
      std::shared_ptr<llama_msgs::srv::UpdateLoRAs::Response> response);

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const GenerateResponse::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleGenerateResponse> goal_handle);
  void handle_accepted(
      const std::shared_ptr<GoalHandleGenerateResponse> goal_handle);

  rclcpp_action::GoalResponse handle_goal_chat_completions(
      const rclcpp_action::GoalUUID &uuid,
      std::shared_ptr<const GenerateChatCompletions::Goal> goal);
  rclcpp_action::CancelResponse handle_cancel_chat_completions(
      const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle);
  void handle_accepted_chat_completions(
      const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle);
};

} // namespace llama_ros

#endif
