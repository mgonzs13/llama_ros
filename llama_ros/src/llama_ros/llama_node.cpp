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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "llama.h"
#include "common.h"
#include "llama_msgs/msg/token_prob.hpp"
#include "llama_msgs/msg/token_prob_array.hpp"
#include "llama_ros/llama_node.hpp"

using namespace llama_ros;
using std::placeholders::_1;
using std::placeholders::_2;

LlamaNode::LlamaNode()
    : rclcpp_lifecycle::LifecycleNode("llama_node"), gpt_params(nullptr) {
  RCLCPP_INFO(this->get_logger(), "%s started", this->get_name());
}

void LlamaNode::create_llama() {
  this->llama = std::make_unique<Llama>(this->gpt_params->params,
                                        this->gpt_params->debug);
}

void LlamaNode::destroy_llama() {
  this->llama.reset();
  this->llama = nullptr;
}

/*
*****************************
*         LIFECYCLE         *
*****************************
*/
rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_configure(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(get_logger(), "[%s] Configuring...", this->get_name());

  if (this->gpt_params == nullptr) {
    this->gpt_params =
        std::make_unique<llama_utils::GptParams>(this->shared_from_this());
    this->gpt_params->declare_params();
  }

  this->gpt_params->get_params();

  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_activate(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(get_logger(), "[%s] Activating...", this->get_name());

  // create llama
  this->create_llama();

  // services
  this->tokenize_service_ = this->create_service<llama_msgs::srv::Tokenize>(
      "tokenize",
      std::bind(&LlamaNode::tokenize_service_callback, this, _1, _2));
  this->generate_embeddings_service_ =
      this->create_service<llama_msgs::srv::GenerateEmbeddings>(
          "generate_embeddings",
          std::bind(&LlamaNode::generate_embeddings_service_callback, this, _1,
                    _2));
  this->format_chat_service_ = this->create_service<llama_msgs::srv::ChatMessages>(
          "format_chat_prompt",
          std::bind(&LlamaNode::format_chat_service_callback, this, _1,
                    _2));

  // generate response action server
  this->goal_handle_ = nullptr;
  this->generate_response_action_server_ =
      rclcpp_action::create_server<GenerateResponse>(
          this, "generate_response",
          std::bind(&LlamaNode::handle_goal, this, _1, _2),
          std::bind(&LlamaNode::handle_cancel, this, _1),
          std::bind(&LlamaNode::handle_accepted, this, _1));

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_deactivate(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(get_logger(), "[%s] Deactivating...", this->get_name());

  this->destroy_llama();

  this->tokenize_service_.reset();
  this->tokenize_service_ = nullptr;

  this->generate_embeddings_service_.reset();
  this->generate_embeddings_service_ = nullptr;

  this->format_chat_service_.reset();
  this->format_chat_service_ = nullptr;

  this->goal_handle_ = nullptr;
  this->generate_response_action_server_.reset();
  this->generate_response_action_server_ = nullptr;

  RCLCPP_INFO(get_logger(), "[%s] Deactivated", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_cleanup(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(get_logger(), "[%s] Cleaning up...", this->get_name());
  RCLCPP_INFO(get_logger(), "[%s] Cleaned up", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_shutdown(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(get_logger(), "[%s] Shutting down...", this->get_name());
  RCLCPP_INFO(get_logger(), "[%s] Shutted down", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

/*
*****************************
*     TOKENIZE SERVICE      *
*****************************
*/
void LlamaNode::tokenize_service_callback(
    const std::shared_ptr<llama_msgs::srv::Tokenize::Request> request,
    std::shared_ptr<llama_msgs::srv::Tokenize::Response> response) {

  response->tokens = this->llama->tokenize(request->prompt, false);
}

/*
*****************************
*    EMBEDDINGS SERVICE     *
*****************************
*/
void LlamaNode::generate_embeddings_service_callback(
    const std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Request> request,
    std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Response> response) {

  auto embeddings =
      this->llama->generate_embeddings(request->prompt, request->normalize);
  response->embeddings = embeddings.embeddings;
  response->n_tokens = embeddings.n_tokens;
}

/*
*****************************
*    FORMAT CHAT SERVICE     *
*****************************
*/
void LlamaNode::format_chat_service_callback(
    const std::shared_ptr<llama_msgs::srv::ChatMessages::Request> request,
    std::shared_ptr<llama_msgs::srv::ChatMessages::Response> response) {

  std::vector<llama_chat_msg> converted_messages;
  for (auto message : request->messages) {
    llama_chat_msg aux;
    aux.role = message.role.c_str();
    aux.content = message.content.c_str();

    converted_messages.push_back(aux);
  }

  std::string formatted_chat = this->llama->format_chat_prompt(converted_messages, request->add_ass);

  response->formatted_prompt = formatted_chat;
}

/*
*****************************
*     GENERATE RESPONSE     *
*****************************
*/
rclcpp_action::GoalResponse
LlamaNode::handle_goal(const rclcpp_action::GoalUUID &uuid,
                       std::shared_ptr<const GenerateResponse::Goal> goal) {
  (void)uuid;
  (void)goal;

  if (this->goal_handle_ != nullptr && this->goal_handle_->is_active()) {
    return rclcpp_action::GoalResponse::REJECT;
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse LlamaNode::handle_cancel(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) {
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "Received request to cancel Llama node");
  this->llama->cancel();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void LlamaNode::handle_accepted(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) {
  this->goal_handle_ = goal_handle;
  std::thread{std::bind(&LlamaNode::execute, this, _1), goal_handle}.detach();
}

bool LlamaNode::goal_empty(std::shared_ptr<const GenerateResponse::Goal> goal) {
  return goal->prompt.size() == 0;
}

void LlamaNode::execute(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) {

  // get goal data
  this->goal_handle_ = goal_handle;
  auto goal = goal_handle->get_goal();
  std::string prompt = goal->prompt;
  std::vector<std::string> stop = goal->stop;
  bool reset = goal_handle->get_goal()->reset;
  auto result = std::make_shared<GenerateResponse::Result>();

  // check if goal is empty
  if (this->goal_empty(goal)) {
    this->goal_handle_->abort(result);
    return;
  }

  if (this->gpt_params->debug) {
    RCLCPP_INFO(this->get_logger(), "Prompt received:\n%s", prompt.c_str());
  }

  // reset llama
  if (reset) {
    this->llama->reset();
  }

  // update sampling params of gpt_params
  auto sampling_config = goal_handle->get_goal()->sampling_config;
  this->gpt_params->update_sampling_params(sampling_config,
                                           this->llama->get_n_vocab(),
                                           this->llama->get_token_eos());

  // call llama
  struct response_output output = this->llama->generate_response(
      prompt, std::bind(&LlamaNode::send_text, this, _1));

  if (output.stop == stop_type::FULL_STOP) {
    auto completion_results = output.completions;

    for (auto completion : completion_results) {
      result->response.text.append(this->llama->detokenize({completion.token}));
      result->response.tokens.push_back(completion.token);

      llama_msgs::msg::TokenProbArray probs_msg;
      for (auto prob : completion.probs) {
        llama_msgs::msg::TokenProb aux;
        aux.token = prob.token;
        aux.probability = prob.probability;
        aux.token_text = this->llama->detokenize({prob.token});
        probs_msg.data.push_back(aux);
      }
      result->response.probs.push_back(probs_msg);
    }
  }

  if (rclcpp::ok()) {

    if (output.stop == stop_type::CANCEL) {
      this->goal_handle_->canceled(result);

    } else if (output.stop == stop_type::ABORT) {
      this->goal_handle_->abort(result);

    } else {
      this->goal_handle_->succeed(result);
    }

    this->goal_handle_ = nullptr;
  }
}

void LlamaNode::send_text(const struct completion_output &completion) {

  if (this->goal_handle_ != nullptr) {
    auto feedback = std::make_shared<GenerateResponse::Feedback>();

    feedback->partial_response.text =
        this->llama->detokenize({completion.token});
    feedback->partial_response.token = completion.token;
    feedback->partial_response.probs.chosen_token = completion.token;

    for (auto prob : completion.probs) {
      llama_msgs::msg::TokenProb aux;
      aux.token = prob.token;
      aux.probability = prob.probability;
      aux.token_text = this->llama->detokenize({prob.token});
      feedback->partial_response.probs.data.push_back(aux);
    }

    this->goal_handle_->publish_feedback(feedback);
  }
}
