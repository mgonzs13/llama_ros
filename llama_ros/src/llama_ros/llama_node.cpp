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

#include "common.h"
#include "llama.h"
#include "llama_msgs/msg/token_prob.hpp"
#include "llama_msgs/msg/token_prob_array.hpp"
#include "llama_ros/llama_node.hpp"

using namespace llama_ros;
using std::placeholders::_1;
using std::placeholders::_2;

LlamaNode::LlamaNode() : rclcpp::Node("llama_node") {

  // load llama
  this->gpt_params.load_params(this);
  this->llama = std::make_shared<Llama>(this->get_logger(), gpt_params.params,
                                        gpt_params.debug);
  this->llama->generate_response(gpt_params.params->prompt, false, nullptr);

  // services
  this->tokenize_service_ = this->create_service<llama_msgs::srv::Tokenize>(
      "tokenize",
      std::bind(&LlamaNode::tokenize_service_callback, this, _1, _2));
  this->generate_embeddings_service_ =
      this->create_service<llama_msgs::srv::GenerateEmbeddings>(
          "generate_embeddings",
          std::bind(&LlamaNode::generate_embeddings_service_callback, this, _1,
                    _2));

  // generate response action server
  this->goal_handle_ = nullptr;
  this->generate_response_action_server_ =
      rclcpp_action::create_server<GenerateResponse>(
          this, "generate_response",
          std::bind(&LlamaNode::handle_goal, this, _1, _2),
          std::bind(&LlamaNode::handle_cancel, this, _1),
          std::bind(&LlamaNode::handle_accepted, this, _1));

  RCLCPP_INFO(this->get_logger(), "Llama Node started");
}

void LlamaNode::tokenize_service_callback(
    const std::shared_ptr<llama_msgs::srv::Tokenize::Request> request,
    std::shared_ptr<llama_msgs::srv::Tokenize::Response> response) {

  response->tokens = this->llama->tokenize(request->prompt, false);
}

void LlamaNode::generate_embeddings_service_callback(
    const std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Request> request,
    std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Response> response) {

  if (this->llama->is_embedding()) {
    response->embeddings =
        this->llama->generate_embeddings(request->prompt, request->normalize);
  }
}

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

void LlamaNode::execute(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) {

  // get goal data
  std::string prompt = goal_handle->get_goal()->prompt;
  bool reset = goal_handle->get_goal()->reset;

  auto result = std::make_shared<GenerateResponse::Result>();
  this->goal_handle_ = goal_handle;

  if (this->gpt_params.debug) {
    RCLCPP_INFO(this->get_logger(), "Prompt received:\n%s", prompt.c_str());
  }

  // reset llama
  if (reset) {
    this->llama->reset();
  }

  // update sampling params
  auto sampling_config = goal_handle->get_goal()->sampling_config;
  this->gpt_params.update_sampling_params(sampling_config,
                                          this->llama->get_n_vocab(),
                                          this->llama->get_token_eos());

  // call llama
  auto completion_results = this->llama->generate_response(
      prompt, true, std::bind(&LlamaNode::send_text, this, _1));

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

  if (rclcpp::ok()) {

    if (this->goal_handle_->is_canceling()) {
      this->goal_handle_->canceled(result);
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
