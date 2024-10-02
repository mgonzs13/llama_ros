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
#include "llama_msgs/msg/lo_ra.hpp"
#include "llama_msgs/msg/token_prob.hpp"
#include "llama_msgs/msg/token_prob_array.hpp"
#include "llama_ros/llama_node.hpp"
#include "llama_utils/llama_params.hpp"

using namespace llama_ros;
using std::placeholders::_1;
using std::placeholders::_2;

LlamaNode::LlamaNode()
    : rclcpp_lifecycle::LifecycleNode("llama_node"), params_declared(false) {}

void LlamaNode::create_llama() {
  this->llama =
      std::make_unique<Llama>(this->params.params, this->params.debug);
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

  if (!this->params_declared) {
    this->params_declared = true;
    llama_utils::declare_llama_params(this->shared_from_this());
  }

  this->params = llama_utils::get_llama_params(this->shared_from_this());
  RCLCPP_INFO(get_logger(), "[%s] Configured", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_activate(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(get_logger(), "[%s] Activating...", this->get_name());

  // create llama
  this->create_llama();

  // embeddings service
  if (this->llama->is_embedding() && !this->llama->is_reranking()) {
    this->generate_embeddings_service_ =
        this->create_service<llama_msgs::srv::GenerateEmbeddings>(
            "generate_embeddings",
            std::bind(&LlamaNode::generate_embeddings_service_callback, this,
                      _1, _2));
  }

  // rerank service
  if (this->llama->is_reranking()) {
    this->rerank_documents_service_ =
        this->create_service<llama_msgs::srv::RerankDocuments>(
            "rerank_documents",
            std::bind(&LlamaNode::rerank_documents_service_callback, this, _1,
                      _2));
  }

  // completion services and action
  if (!this->llama->is_embedding() && !this->llama->is_reranking()) {
    this->tokenize_service_ = this->create_service<llama_msgs::srv::Tokenize>(
        "tokenize",
        std::bind(&LlamaNode::tokenize_service_callback, this, _1, _2));
    this->detokenize_service_ =
        this->create_service<llama_msgs::srv::Detokenize>(
            "detokenize",
            std::bind(&LlamaNode::detokenize_service_callback, this, _1, _2));

    this->format_chat_service_ =
        this->create_service<llama_msgs::srv::FormatChatMessages>(
            "format_chat_prompt",
            std::bind(&LlamaNode::format_chat_service_callback, this, _1, _2));

    this->list_loras_service_ =
        this->create_service<llama_msgs::srv::ListLoRAs>(
            "list_loras",
            std::bind(&LlamaNode::list_loras_service_callback, this, _1, _2));
    this->update_loras_service_ =
        this->create_service<llama_msgs::srv::UpdateLoRAs>(
            "update_loras",
            std::bind(&LlamaNode::update_loras_service_callback, this, _1, _2));

    // generate response action server
    this->goal_handle_ = nullptr;
    this->generate_response_action_server_ =
        rclcpp_action::create_server<GenerateResponse>(
            this, "generate_response",
            std::bind(&LlamaNode::handle_goal, this, _1, _2),
            std::bind(&LlamaNode::handle_cancel, this, _1),
            std::bind(&LlamaNode::handle_accepted, this, _1));
  }

  RCLCPP_INFO(get_logger(), "[%s] Activated", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_deactivate(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(get_logger(), "[%s] Deactivating...", this->get_name());

  this->destroy_llama();

  if (this->llama->is_embedding() && !this->llama->is_reranking()) {
    this->generate_embeddings_service_.reset();
    this->generate_embeddings_service_ = nullptr;
  }

  if (this->llama->is_reranking()) {
    this->rerank_documents_service_.reset();
    this->rerank_documents_service_ = nullptr;
  }

  if (!this->llama->is_embedding() && !this->llama->is_reranking()) {
    this->tokenize_service_.reset();
    this->tokenize_service_ = nullptr;

    this->detokenize_service_.reset();
    this->detokenize_service_ = nullptr;

    this->format_chat_service_.reset();
    this->format_chat_service_ = nullptr;

    this->list_loras_service_.reset();
    this->list_loras_service_ = nullptr;

    this->update_loras_service_.reset();
    this->update_loras_service_ = nullptr;

    this->goal_handle_ = nullptr;
    this->generate_response_action_server_.reset();
    this->generate_response_action_server_ = nullptr;
  }

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

  response->tokens = this->llama->tokenize(request->text, false);
}

void LlamaNode::detokenize_service_callback(
    const std::shared_ptr<llama_msgs::srv::Detokenize::Request> request,
    std::shared_ptr<llama_msgs::srv::Detokenize::Response> response) {

  std::vector<llama_token> tokens;
  for (auto t : request->tokens) {
    tokens.push_back(t);
  }

  response->text = this->llama->detokenize(tokens);
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
*         RERANKING         *
*****************************
*/
void LlamaNode::rerank_documents_service_callback(
    const std::shared_ptr<llama_msgs::srv::RerankDocuments::Request> request,
    std::shared_ptr<llama_msgs::srv::RerankDocuments::Response> response) {

  response->scores =
      this->llama->rank_documents(request->query, request->documents);
}

/*
*****************************
*    FORMAT CHAT SERVICE    *
*****************************
*/
void LlamaNode::format_chat_service_callback(
    const std::shared_ptr<llama_msgs::srv::FormatChatMessages::Request> request,
    std::shared_ptr<llama_msgs::srv::FormatChatMessages::Response> response) {

  std::vector<struct llama_chat_msg> converted_messages;
  for (auto message : request->messages) {
    struct llama_chat_msg aux;
    aux.role = message.role.c_str();
    aux.content = message.content.c_str();

    converted_messages.push_back(aux);
  }

  std::string formatted_chat =
      this->llama->format_chat_prompt(converted_messages, request->add_ass);

  response->formatted_prompt = formatted_chat;
}

/*
*******************************
*            LORAS            *
*******************************
*/
void LlamaNode::list_loras_service_callback(
    const std::shared_ptr<llama_msgs::srv::ListLoRAs::Request> request,
    std::shared_ptr<llama_msgs::srv::ListLoRAs::Response> response) {

  (void)request;

  auto loras = this->llama->list_loras();

  for (auto lora : loras) {

    llama_msgs::msg::LoRA lora_msg;
    lora_msg.id = lora.id;
    lora_msg.path = lora.path;
    lora_msg.scale = lora.scale;

    response->loras.push_back(lora_msg);
  }
}

void LlamaNode::update_loras_service_callback(
    const std::shared_ptr<llama_msgs::srv::UpdateLoRAs::Request> request,
    std::shared_ptr<llama_msgs::srv::UpdateLoRAs::Response> response) {

  (void)response;

  std::vector<struct lora> loras;

  for (auto lora_msg : request->loras) {

    struct lora lora_aux;
    lora_aux.id = lora_msg.id;
    lora_aux.path = lora_msg.path;
    lora_aux.scale = lora_msg.scale;

    loras.push_back(lora_aux);
  }

  this->llama->update_loras(loras);
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

  if (this->params.debug) {
    RCLCPP_INFO(this->get_logger(), "Prompt received:\n%s", prompt.c_str());
  }

  // reset llama
  if (reset) {
    this->llama->reset();
  }

  // update sampling params of gpt_params
  auto sampling_config = goal_handle->get_goal()->sampling_config;
  auto sparams = llama_utils::parse_sampling_params(sampling_config,
                                                    this->llama->get_n_vocab());

  // call llama
  struct response_output output = this->llama->generate_response(
      prompt, sparams, std::bind(&LlamaNode::send_text, this, _1));

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
