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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "chat.h"
#include "common.h"
#include "llama.h"

#include "llama_msgs/msg/lo_ra.hpp"
#include "llama_msgs/msg/logit_bias.hpp"
#include "llama_msgs/msg/sampling_config.hpp"
#include "llama_msgs/msg/token_prob.hpp"
#include "llama_msgs/msg/token_prob_array.hpp"
#include "llama_ros/llama.hpp"
#include "llama_ros/llama_node.hpp"
#include "llama_utils/chat_utils.hpp"
#include "llama_utils/llama_params.hpp"

using namespace llama_ros;
using std::placeholders::_1;
using std::placeholders::_2;

LlamaNode::LlamaNode()
    : rclcpp_lifecycle::LifecycleNode("llama_node"), params_declared(false) {}

void LlamaNode::create_llama() {
  this->llama =
      std::make_unique<Llama>(this->params.params, this->params.system_prompt);

  run_loop_thread_ = std::thread([this]() {
    try {
      this->llama->run_loop();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Exception in run_loop: %s", e.what());
    } catch (...) {
      RCLCPP_ERROR(this->get_logger(), "Unknown exception in run_loop");
    }
  });
}

void LlamaNode::destroy_llama() {
  if (this->llama) {
    this->llama->cancel();
  }
  
  if (run_loop_thread_.joinable()) {
    try {
      run_loop_thread_.join();
    } catch (const std::exception& e) {
      RCLCPP_ERROR(this->get_logger(), "Exception while joining run_loop thread: %s", e.what());
    }
  }
  
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

  RCLCPP_INFO(this->get_logger(), "[%s] Configuring...", this->get_name());

  if (!this->params_declared) {
    this->params_declared = true;
    llama_utils::declare_llama_params(this->shared_from_this());
  }

  this->params = llama_utils::get_llama_params(this->shared_from_this());
  RCLCPP_INFO(this->get_logger(), "[%s] Configured", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_activate(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(this->get_logger(), "[%s] Activating...", this->get_name());

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
    // get metadata service
    this->get_metadata_service_ =
        this->create_service<llama_msgs::srv::GetMetadata>(
            "get_metadata",
            std::bind(&LlamaNode::get_metadata_service_callback, this, _1, _2));

    this->tokenize_service_ = this->create_service<llama_msgs::srv::Tokenize>(
        "tokenize",
        std::bind(&LlamaNode::tokenize_service_callback, this, _1, _2));
    this->detokenize_service_ =
        this->create_service<llama_msgs::srv::Detokenize>(
            "detokenize",
            std::bind(&LlamaNode::detokenize_service_callback, this, _1, _2));

    this->list_loras_service_ =
        this->create_service<llama_msgs::srv::ListLoRAs>(
            "list_loras",
            std::bind(&LlamaNode::list_loras_service_callback, this, _1, _2));
    this->update_loras_service_ =
        this->create_service<llama_msgs::srv::UpdateLoRAs>(
            "update_loras",
            std::bind(&LlamaNode::update_loras_service_callback, this, _1, _2));

    // generate response action server
    this->generate_response_action_server_ =
        rclcpp_action::create_server<GenerateResponse>(
            this, "generate_response",
            std::bind(&LlamaNode::handle_goal, this, _1, _2),
            std::bind(&LlamaNode::handle_cancel, this, _1),
            std::bind(&LlamaNode::handle_accepted, this, _1));

    this->generate_chat_completions_action_server_ =
        rclcpp_action::create_server<GenerateChatCompletions>(
            this, "generate_chat_completions",
            std::bind(&LlamaNode::handle_goal_chat_completions, this, _1, _2),
            std::bind(&LlamaNode::handle_cancel_chat_completions, this, _1),
            std::bind(&LlamaNode::handle_accepted_chat_completions, this, _1));
  }

  RCLCPP_INFO(this->get_logger(), "[%s] Activated", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_deactivate(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(this->get_logger(), "[%s] Deactivating...", this->get_name());

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
    this->get_metadata_service_.reset();
    this->get_metadata_service_ = nullptr;

    this->tokenize_service_.reset();
    this->tokenize_service_ = nullptr;

    this->detokenize_service_.reset();
    this->detokenize_service_ = nullptr;

    this->list_loras_service_.reset();
    this->list_loras_service_ = nullptr;

    this->update_loras_service_.reset();
    this->update_loras_service_ = nullptr;

    this->generate_response_action_server_.reset();
    this->generate_response_action_server_ = nullptr;

    this->generate_chat_completions_action_server_.reset();
    this->generate_chat_completions_action_server_ = nullptr;
  }

  RCLCPP_INFO(this->get_logger(), "[%s] Deactivated", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_cleanup(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(this->get_logger(), "[%s] Cleaning up...", this->get_name());
  RCLCPP_INFO(this->get_logger(), "[%s] Cleaned up", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
LlamaNode::on_shutdown(const rclcpp_lifecycle::State &) {

  RCLCPP_INFO(this->get_logger(), "[%s] Shutting down...", this->get_name());
  RCLCPP_INFO(this->get_logger(), "[%s] Shutted down", this->get_name());

  return rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::
      CallbackReturn::SUCCESS;
}

/*
*****************************
*         METADATA          *
*****************************
*/
void LlamaNode::get_metadata_service_callback(
    const std::shared_ptr<llama_msgs::srv::GetMetadata::Request> request,
    std::shared_ptr<llama_msgs::srv::GetMetadata::Response> response) {

  (void)request;

  llama_ros::Metadata metadata = this->llama->get_metadata();

  llama_msgs::msg::Metadata metadata_msg;

  // general
  metadata_msg.general.architecture = metadata.general.architecture;
  metadata_msg.general.quantization_version =
      metadata.general.quantization_version;
  metadata_msg.general.alignment = metadata.general.alignment;

  metadata_msg.general.name = metadata.general.name;
  metadata_msg.general.author = metadata.general.author;
  metadata_msg.general.version = metadata.general.version;
  metadata_msg.general.organization = metadata.general.organization;

  metadata_msg.general.basename = metadata.general.basename;
  metadata_msg.general.finetune = metadata.general.finetune;
  metadata_msg.general.description = metadata.general.description;
  metadata_msg.general.quantized_by = metadata.general.quantized_by;
  metadata_msg.general.size_label = metadata.general.size_label;

  metadata_msg.general.license = metadata.general.license;
  metadata_msg.general.license_name = metadata.general.license_name;
  metadata_msg.general.license_link = metadata.general.license_link;

  metadata_msg.general.url = metadata.general.url;
  metadata_msg.general.repo_url = metadata.general.repo_url;
  metadata_msg.general.doi = metadata.general.doi;
  metadata_msg.general.uuid = metadata.general.uuid;

  metadata_msg.general.file_type = metadata.general.file_type;

  // model
  metadata_msg.model.context_length = metadata.model.context_length;
  metadata_msg.model.embedding_length = metadata.model.embedding_length;
  metadata_msg.model.block_count = metadata.model.block_count;
  metadata_msg.model.feed_forward_length = metadata.model.feed_forward_length;

  metadata_msg.model.use_parallel_residual =
      metadata.model.use_parallel_residual;
  metadata_msg.model.tensor_data_layout = metadata.model.tensor_data_layout;

  metadata_msg.model.expert_count = metadata.model.expert_count;
  metadata_msg.model.expert_used_count = metadata.model.expert_used_count;

  // attention
  metadata_msg.model.attention.head_count = metadata.model.attention.head_count;
  metadata_msg.model.attention.head_count_kv =
      metadata.model.attention.head_count_kv;

  metadata_msg.model.attention.max_alibi_bias =
      metadata.model.attention.max_alibi_bias;
  metadata_msg.model.attention.clamp_kqv = metadata.model.attention.clamp_kqv;

  metadata_msg.model.attention.layer_norm_epsilon =
      metadata.model.attention.layer_norm_epsilon;
  metadata_msg.model.attention.layer_norm_rms_epsilon =
      metadata.model.attention.layer_norm_rms_epsilon;

  metadata_msg.model.attention.key_length = metadata.model.attention.key_length;
  metadata_msg.model.attention.value_length =
      metadata.model.attention.value_length;

  // rope
  metadata_msg.model.rope.dimension_count = metadata.model.rope.dimension_count;
  metadata_msg.model.rope.freq_base = metadata.model.rope.freq_base;

  metadata_msg.model.rope.scaling_type = metadata.model.rope.scaling_type;
  metadata_msg.model.rope.scaling_factor = metadata.model.rope.scaling_factor;
  metadata_msg.model.rope.scaling_original_context_length =
      metadata.model.rope.scaling_original_context_length;
  metadata_msg.model.rope.scaling_finetuned =
      metadata.model.rope.scaling_finetuned;

  // tokenizer
  metadata_msg.tokenizer.model = metadata.tokenizer.model;

  metadata_msg.tokenizer.bos_token_id = metadata.tokenizer.bos_token_id;
  metadata_msg.tokenizer.eos_token_id = metadata.tokenizer.eos_token_id;
  metadata_msg.tokenizer.unknown_token_id = metadata.tokenizer.unknown_token_id;
  metadata_msg.tokenizer.padding_token_id = metadata.tokenizer.padding_token_id;
  metadata_msg.tokenizer.separator_token_id =
      metadata.tokenizer.separator_token_id;

  metadata_msg.tokenizer.add_bos_token = metadata.tokenizer.add_bos_token;
  metadata_msg.tokenizer.chat_template = metadata.tokenizer.chat_template;

  response->metadata = metadata_msg;
}

/*
*****************************
*     TOKENIZE SERVICE      *
*****************************
*/
void LlamaNode::tokenize_service_callback(
    const std::shared_ptr<llama_msgs::srv::Tokenize::Request> request,
    std::shared_ptr<llama_msgs::srv::Tokenize::Response> response) {

  response->tokens = this->llama->tokenize(request->text, false, 2);
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
  RCLCPP_INFO(this->get_logger(), "Generating embeddings");

  auto result = this->llama->generate_embeddings(request->prompt);
  if (result.is_error()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to generate embeddings: %s", result.error().c_str());
    return;
  }

  auto embeddings = result.value();
  std::vector<std::vector<float>> data = embeddings.embeddings;
  response->embeddings = std::vector<float>();
  for (const auto &vec : data) {
    response->embeddings.insert(response->embeddings.end(), vec.begin(), vec.end());
  }
  response->n_tokens = embeddings.n_tokens;

  RCLCPP_INFO(this->get_logger(), "Embeddings generated");
}

/*
*****************************
*         RERANKING         *
*****************************
*/
void LlamaNode::rerank_documents_service_callback(
    const std::shared_ptr<llama_msgs::srv::RerankDocuments::Request> request,
    std::shared_ptr<llama_msgs::srv::RerankDocuments::Response> response) {
  RCLCPP_INFO(this->get_logger(), "Generating reranking");

  auto result = this->llama->rank_documents(request->query, request->documents);

  if (result.is_error()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to generate reranking: %s", result.error().c_str());
    return;
  }

  auto reranks = result.value();
  response->scores = std::vector<float>(request->documents.size(), -1e6);
  for (uint32_t i = 0; i < reranks.size(); i++) {
    const auto &rerank = reranks[i];
    response->scores[i] = rerank.score;
  }

  RCLCPP_INFO(this->get_logger(), "Reranking finished");
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

  std::vector<struct LoRA> loras;

  for (auto lora_msg : request->loras) {

    struct LoRA lora_aux;
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

  RCLCPP_INFO(this->get_logger(), "Received goal to generate response");

  ServerSlot *slot = this->llama->wait_for_available_slot();
  if (slot == nullptr) {
    RCLCPP_ERROR(this->get_logger(), "No slot available");
    return rclcpp_action::GoalResponse::REJECT;
  }
  slot->goal_id = llama_utils::uuid_to_int32(uuid);
  RCLCPP_INFO(this->get_logger(), "Assigned slot %d to goal %lu",
              slot->id, slot->goal_id);

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
  int slot_gid = llama_utils::uuid_to_int32(goal_handle->get_goal_id());
  std::thread{std::bind(&LlamaNode::execute, this, _1, _2), goal_handle, slot_gid}.detach();
}

bool LlamaNode::goal_empty(std::shared_ptr<const GenerateResponse::Goal> goal) {
  return goal->prompt.size() == 0;
}

void LlamaNode::execute(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle,
    int slot_id) {
  auto goal = goal_handle->get_goal();
  auto response = std::make_shared<GenerateResponse::Result>();

  // Validate goal
  if (this->goal_empty(goal)) {
    goal_handle->abort(response);
    return;
  }

  // Check if llama is initialized
  if (!this->llama) {
    RCLCPP_ERROR(this->get_logger(), "Llama is not initialized");
    goal_handle->abort(response);
    return;
  }

  // Prepare request using utility
  auto context = llama_utils::prepare_completion_call(goal, this->llama.get());

  // Execute via Llama
  auto result = this->llama->generate_response(
      slot_id, context.prompt, context.sparams,
      [this, goal_handle, slot_id](const struct CompletionOutput &completion, ServerSlot *) {
        this->send_text(completion, goal_handle, slot_id);
      },
      context.stop, context.reset);

  // Handle result
  if (result.is_error()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to generate response: %s",
                 result.error().c_str());
    goal_handle->abort(response);
    return;
  }

  // Convert result using utility
  *response =
      llama_utils::generate_completion_result(result.value(), this->llama.get());

  // Publish based on stop type
  if (!rclcpp::ok() || !goal_handle) {
    return;
  }

  switch (result.value().stop) {
    case StopType::CANCEL:
      goal_handle->canceled(response);
      break;
    case StopType::ABORT:
      goal_handle->abort(response);
      break;
    default:
      goal_handle->succeed(response);
      break;
  }
}

void LlamaNode::send_text(const struct CompletionOutput &completion, 
                          const std::shared_ptr<GoalHandleGenerateResponse> &goal_handle, int slot_id) {
  (void)slot_id;
                            if (goal_handle && this->llama) {
    auto feedback = std::make_shared<GenerateResponse::Feedback>();
    *feedback = llama_utils::create_completion_feedback(completion, this->llama.get());
    goal_handle->publish_feedback(feedback);
  }
}

/*
*****************************
*     GENERATE CHAT         *
*****************************
*/

rclcpp_action::GoalResponse LlamaNode::handle_goal_chat_completions(
    const rclcpp_action::GoalUUID &uuid,
    std::shared_ptr<const GenerateChatCompletions::Goal> goal) {
  (void)uuid;
  (void)goal;

  RCLCPP_INFO(this->get_logger(), "Received goal to generate response");

  RCLCPP_INFO(this->get_logger(), "Waiting for available slot");
  ServerSlot *slot = this->llama->wait_for_available_slot();
  if (slot == nullptr) {
    RCLCPP_ERROR(this->get_logger(), "No slot available");
    return rclcpp_action::GoalResponse::REJECT;
  }
  slot->goal_id = llama_utils::uuid_to_int32(uuid);
  RCLCPP_INFO(this->get_logger(), "Assigned slot %d to goal %lu",
              slot->id, slot->goal_id);

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse LlamaNode::handle_cancel_chat_completions(
    const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle) {
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "Received request to cancel Llama node");
  this->llama->cancel();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void LlamaNode::handle_accepted_chat_completions(
    const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle) {
  int slot_gid = llama_utils::uuid_to_int32(goal_handle->get_goal_id());
  std::thread{std::bind(&LlamaNode::execute_chat_completions, this, _1, _2), goal_handle, slot_gid}.detach();
}

bool LlamaNode::goal_empty_chat_completions(
    std::shared_ptr<const GenerateChatCompletions::Goal> goal) {
  return goal->messages.size() == 0;
}

void LlamaNode::execute_chat_completions(
    const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle,
    int slot_gid) {
  auto goal = goal_handle->get_goal();
  auto parsed_result = std::make_shared<GenerateChatCompletions::Result>();

  // Validate goal
  if (this->goal_empty_chat_completions(goal)) {
    goal_handle->abort(parsed_result);
    return;
  }

  // Check if llama is initialized
  if (!this->llama) {
    RCLCPP_ERROR(this->get_logger(), "Llama is not initialized");
    goal_handle->abort(parsed_result);
    return;
  }

  // Prepare request using utility
  auto chat_context =
      llama_utils::prepare_chat_completions_call(goal, this->llama.get());

  // Execute via Llama
  auto result_data = this->llama->generate_chat_response(
      slot_gid, chat_context,
      [this, goal_handle, slot_gid](const struct CompletionOutput &completion, ServerSlot *) {
        this->send_text_chat_completions(completion, goal_handle, slot_gid);
      });

  // Handle result
  if (result_data.is_error()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to generate response: %s",
                 result_data.error().c_str());
    goal_handle->abort(parsed_result);
    return;
  }

  // Convert result using utility
  *parsed_result = llama_utils::generate_chat_completions_result(result_data.value());

  // Publish based on stop type
  if (!rclcpp::ok() || !goal_handle) {
    return;
  }

  switch (result_data.value().stop) {
    case StopType::CANCEL:
      goal_handle->canceled(parsed_result);
      break;
    case StopType::ABORT:
      goal_handle->abort(parsed_result);
      break;
    default:
      goal_handle->succeed(parsed_result);
      break;
  }
}

void LlamaNode::send_text_chat_completions(
    const struct CompletionOutput &completion,
    const std::shared_ptr<GoalHandleGenerateChatCompletions> &goal_handle, int slot_id) {
  if (this->llama && goal_handle) {
    // Get slot_id from goal_handle's goal
    auto goal = goal_handle->get_goal();
    auto slot = this->llama->get_slot_by_gid(slot_id);

    llama_ros::ServerTaskResultCompletionPartial response_result;

    response_result.oaicompat_model = this->llama->get_metadata().general.name;
    response_result.oaicompat_cmpl_id = "chatcmplid-0";
    response_result.build_info =
        "b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT;
    response_result.content = this->llama->detokenize({completion.token});
    response_result.stop = llama_ros::StopType::NO_STOP;
    response_result.post_sampling_probs = false;

    slot->update_chat_msg(response_result.oaicompat_msg_diffs);

    auto feedbacks = llama_utils::generate_chat_completions_feedback(
        response_result, response_result.oaicompat_msg_diffs);

    for (auto &feedback : feedbacks) {
      goal_handle->publish_feedback(
          std::make_shared<GenerateChatCompletions::Feedback>(feedback));
    }
  }
}