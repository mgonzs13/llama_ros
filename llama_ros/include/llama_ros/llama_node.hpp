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
#include <thread>

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

/**
 * @brief Represents a ROS 2 lifecycle node for managing Llama operations.
 *
 * This class provides services and actions for interacting with the llama.cpp,
 * including generating responses, chat completions, and managing metadata.
 */
class LlamaNode : public rclcpp_lifecycle::LifecycleNode {

  /**
   * @brief Action definition for generating responses.
   */
  using GenerateResponse = llama_msgs::action::GenerateResponse;

  /**
   * @brief Goal handle for the GenerateResponse action.
   */
  using GoalHandleGenerateResponse =
      rclcpp_action::ServerGoalHandle<GenerateResponse>;

  /**
   * @brief Action definition for generating chat completions.
   */
  using GenerateChatCompletions = llama_msgs::action::GenerateChatCompletions;

  /**
   * @brief Goal handle for the GenerateChatCompletions action.
   */
  using GoalHandleGenerateChatCompletions =
      rclcpp_action::ServerGoalHandle<GenerateChatCompletions>;

public:
  /**
   * @brief Constructs a new LlamaNode instance.
   *
   * Initializes the node and sets up the necessary services and actions.
   */
  LlamaNode();

  /**
   * @brief Callback for the "configure" lifecycle transition.
   *
   * This method is called when the node transitions to the "configured" state.
   *
   * @param state The current lifecycle state.
   * @return The result of the transition, indicating success or failure.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_configure(const rclcpp_lifecycle::State &state);

  /**
   * @brief Callback for the "activate" lifecycle transition.
   *
   * This method is called when the node transitions to the "active" state.
   *
   * @param state The current lifecycle state.
   * @return The result of the transition, indicating success or failure.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_activate(const rclcpp_lifecycle::State &state);

  /**
   * @brief Callback for the "deactivate" lifecycle transition.
   *
   * This method is called when the node transitions to the "inactive" state.
   *
   * @param state The current lifecycle state.
   * @return The result of the transition, indicating success or failure.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_deactivate(const rclcpp_lifecycle::State &state);

  /**
   * @brief Callback for the "cleanup" lifecycle transition.
   *
   * This method is called when the node transitions to the "cleaned up" state.
   *
   * @param state The current lifecycle state.
   * @return The result of the transition, indicating success or failure.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_cleanup(const rclcpp_lifecycle::State &state);

  /**
   * @brief Callback for the "shutdown" lifecycle transition.
   *
   * This method is called when the node transitions to the "shutdown" state.
   *
   * @param state The current lifecycle state.
   * @return The result of the transition, indicating success or failure.
   */
  rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
  on_shutdown(const rclcpp_lifecycle::State &state);

protected:
  /**
   * @brief Pointer to the Llama instance.
   *
   * This unique pointer manages the lifecycle of the Llama object used for
   * model operations.
   */
  std::unique_ptr<Llama> llama;

  /**
   * @brief Indicates whether the parameters have been declared.
   *
   * This boolean flag is used to track if the node's parameters have been
   * properly declared.
   */
  bool params_declared;

  /**
   * @brief Parameters for configuring the Llama model.
   *
   * This structure contains the configuration options for initializing and
   * managing the Llama model.
   */
  struct llama_utils::LlamaParams params;

  /**
   * @brief Creates and initializes the Llama instance.
   *
   * This virtual method is responsible for creating the Llama object and
   * setting up its configuration.
   */
  virtual void create_llama();

  /**
   * @brief Destroys the Llama instance.
   *
   * This method is responsible for cleaning up and releasing resources
   * associated with the Llama object.
   */
  void destroy_llama();

  /**
   * @brief Checks if the GenerateResponse goal is empty.
   *
   * This method validates whether the provided goal for the GenerateResponse
   * action is empty.
   *
   * @param goal A shared pointer to the GenerateResponse goal.
   * @return True if the goal is empty, false otherwise.
   */
  virtual bool goal_empty(std::shared_ptr<const GenerateResponse::Goal> goal);

  /**
   * @brief Executes the GenerateResponse action.
   *
   * This method handles the execution of the GenerateResponse action for the
   * provided goal handle.
   *
   * @param goal_handle A shared pointer to the goal handle for the
   * GenerateResponse action.
   */
  virtual void
  execute(const std::shared_ptr<GoalHandleGenerateResponse> goal_handle, int slot_id);

  /**
   * @brief Sends the generated text response.
   *
   * This method sends the generated text response to the client.
   *
   * @param completion The completion output containing the generated text.
   * @param goal_handle The goal handle for the GenerateResponse action.
   */
  void send_text(const struct CompletionOutput &completion, 
                 const std::shared_ptr<GoalHandleGenerateResponse> &goal_handle, int slot_id);

  /**
   * @brief Checks if the GenerateChatCompletions goal is empty.
   *
   * This method validates whether the provided goal for the
   * GenerateChatCompletions action is empty.
   *
   * @param goal A shared pointer to the GenerateChatCompletions goal.
   * @return True if the goal is empty, false otherwise.
   */
  virtual bool goal_empty_chat_completions(
      std::shared_ptr<const GenerateChatCompletions::Goal> goal);

  /**
   * @brief Executes the GenerateChatCompletions action.
   *
   * This method handles the execution of the GenerateChatCompletions action for
   * the provided goal handle.
   *
   * @param goal_handle A shared pointer to the goal handle for the
   * GenerateChatCompletions action.
   */
  virtual void execute_chat_completions(
      const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle, int slot_id);

  /**
   * @brief Sends the generated chat completion response.
   *
   * This method sends the generated chat completion response to the client.
   *
   * @param completion The completion output containing the generated chat
   * response.
   * @param goal_handle The goal handle for the GenerateChatCompletions action.
   */
  void send_text_chat_completions(const struct CompletionOutput &completion,
                                   const std::shared_ptr<GoalHandleGenerateChatCompletions> &goal_handle, int slot_id);

  /**
   * @brief Thread running the Llama run_loop.
   *
   * This thread continuously processes slots in the background.
   */
  std::thread run_loop_thread_;

private:
  /**
   * @brief Service for retrieving metadata.
   *
   * This service allows clients to request metadata information from the Llama
   * model.
   */
  rclcpp::Service<llama_msgs::srv::GetMetadata>::SharedPtr
      get_metadata_service_;

  /**
   * @brief Service for tokenizing text.
   *
   * This service allows clients to tokenize input text into a sequence of
   * tokens.
   */
  rclcpp::Service<llama_msgs::srv::Tokenize>::SharedPtr tokenize_service_;

  /**
   * @brief Service for detokenizing tokens.
   *
   * This service allows clients to convert a sequence of tokens back into text.
   */
  rclcpp::Service<llama_msgs::srv::Detokenize>::SharedPtr detokenize_service_;

  /**
   * @brief Service for generating embeddings.
   *
   * This service allows clients to generate embeddings for input text or
   * tokens.
   */
  rclcpp::Service<llama_msgs::srv::GenerateEmbeddings>::SharedPtr
      generate_embeddings_service_;

  /**
   * @brief Service for reranking documents.
   *
   * This service allows clients to rerank a set of documents based on a query.
   */
  rclcpp::Service<llama_msgs::srv::RerankDocuments>::SharedPtr
      rerank_documents_service_;

  /**
   * @brief Service for listing available LoRA (Low-Rank Adaptation) models.
   *
   * This service allows clients to retrieve a list of available LoRA models.
   */
  rclcpp::Service<llama_msgs::srv::ListLoRAs>::SharedPtr list_loras_service_;

  /**
   * @brief Service for updating LoRA (Low-Rank Adaptation) models.
   *
   * This service allows clients to update the LoRA models used by the Llama
   * instance.
   */
  rclcpp::Service<llama_msgs::srv::UpdateLoRAs>::SharedPtr
      update_loras_service_;

  /**
   * @brief Action server for generating responses.
   *
   * This action server handles requests for generating text responses based on
   * input prompts.
   */
  rclcpp_action::Server<GenerateResponse>::SharedPtr
      generate_response_action_server_;

  /**
   * @brief Action server for generating chat completions.
   *
   * This action server handles requests for generating chat completions based
   * on input prompts.
   */
  rclcpp_action::Server<GenerateChatCompletions>::SharedPtr
      generate_chat_completions_action_server_;

  /**
   * @brief Handles a new goal request for the GenerateResponse action.
   *
   * This method is called when a new goal is received for the GenerateResponse
   * action.
   *
   * @param uuid The unique identifier for the goal.
   * @param goal A shared pointer to the GenerateResponse goal.
   * @return A GoalResponse indicating whether the goal is accepted or rejected.
   */

  /**
   * @brief Callback for the GetMetadata service.
   *
   * This service retrieves metadata information from the Llama model.
   *
   * @param request The request object containing the metadata query.
   * @param response The response object to populate with metadata information.
   */
  void get_metadata_service_callback(
      const std::shared_ptr<llama_msgs::srv::GetMetadata::Request> request,
      std::shared_ptr<llama_msgs::srv::GetMetadata::Response> response);

  /**
   * @brief Callback for the Tokenize service.
   *
   * This service tokenizes input text into a sequence of tokens.
   *
   * @param request The request object containing the text to tokenize.
   * @param response The response object to populate with the tokenized output.
   */
  void tokenize_service_callback(
      const std::shared_ptr<llama_msgs::srv::Tokenize::Request> request,
      std::shared_ptr<llama_msgs::srv::Tokenize::Response> response);

  /**
   * @brief Callback for the Detokenize service.
   *
   * This service converts a sequence of tokens back into text.
   *
   * @param request The request object containing the tokens to detokenize.
   * @param response The response object to populate with the detokenized text.
   */
  void detokenize_service_callback(
      const std::shared_ptr<llama_msgs::srv::Detokenize::Request> request,
      std::shared_ptr<llama_msgs::srv::Detokenize::Response> response);

  /**
   * @brief Callback for the GenerateEmbeddings service.
   *
   * This service generates embeddings for input text or tokens.
   *
   * @param request The request object containing the input for embedding
   * generation.
   * @param response The response object to populate with the generated
   * embeddings.
   */
  void generate_embeddings_service_callback(
      const std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Request>
          request,
      std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Response> response);

  /**
   * @brief Callback for the RerankDocuments service.
   *
   * This service reranks a set of documents based on a query.
   *
   * @param request The request object containing the query and documents to
   * rerank.
   * @param response The response object to populate with the reranked
   * documents.
   */
  void rerank_documents_service_callback(
      const std::shared_ptr<llama_msgs::srv::RerankDocuments::Request> request,
      std::shared_ptr<llama_msgs::srv::RerankDocuments::Response> response);

  /**
   * @brief Callback for the ListLoRAs service.
   *
   * This service retrieves a list of available LoRA (Low-Rank Adaptation)
   * models.
   *
   * @param request The request object for listing LoRA models.
   * @param response The response object to populate with the list of LoRA
   * models.
   */
  void list_loras_service_callback(
      const std::shared_ptr<llama_msgs::srv::ListLoRAs::Request> request,
      std::shared_ptr<llama_msgs::srv::ListLoRAs::Response> response);

  /**
   * @brief Callback for the UpdateLoRAs service.
   *
   * This service updates the LoRA (Low-Rank Adaptation) models used by the
   * Llama instance.
   *
   * @param request The request object containing the update details for LoRA
   * models.
   * @param response The response object to populate with the update status.
   */
  void update_loras_service_callback(
      const std::shared_ptr<llama_msgs::srv::UpdateLoRAs::Request> request,
      std::shared_ptr<llama_msgs::srv::UpdateLoRAs::Response> response);

  /**
   * @brief Handles a new goal request for the GenerateResponse action.
   *
   * This method is called when a new goal is received for the GenerateResponse
   * action.
   *
   * @param uuid The unique identifier for the goal.
   * @param goal A shared pointer to the GenerateResponse goal.
   * @return A GoalResponse indicating whether the goal is accepted or rejected.
   */
  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const GenerateResponse::Goal> goal);

  /**
   * @brief Handles a cancel request for the GenerateResponse action.
   *
   * This method is called when a client requests to cancel an ongoing
   * GenerateResponse action.
   *
   * @param goal_handle A shared pointer to the goal handle for the
   * GenerateResponse action.
   * @return A CancelResponse indicating whether the cancel request is accepted
   * or rejected.
   */
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleGenerateResponse> goal_handle);

  /**
   * @brief Handles an accepted goal for the GenerateResponse action.
   *
   * This method is called when a goal for the GenerateResponse action is
   * accepted.
   *
   * @param goal_handle A shared pointer to the goal handle for the
   * GenerateResponse action.
   */
  void handle_accepted(
      const std::shared_ptr<GoalHandleGenerateResponse> goal_handle);

  /**
   * @brief Handles a new goal request for the GenerateChatCompletions action.
   *
   * This method is called when a new goal is received for the
   * GenerateChatCompletions action.
   *
   * @param uuid The unique identifier for the goal.
   * @param goal A shared pointer to the GenerateChatCompletions goal.
   * @return A GoalResponse indicating whether the goal is accepted or rejected.
   */
  rclcpp_action::GoalResponse handle_goal_chat_completions(
      const rclcpp_action::GoalUUID &uuid,
      std::shared_ptr<const GenerateChatCompletions::Goal> goal);

  /**
   * @brief Handles a cancel request for the GenerateChatCompletions action.
   *
   * This method is called when a client requests to cancel an ongoing
   * GenerateChatCompletions action.
   *
   * @param goal_handle A shared pointer to the goal handle for the
   * GenerateChatCompletions action.
   * @return A CancelResponse indicating whether the cancel request is accepted
   * or rejected.
   */
  rclcpp_action::CancelResponse handle_cancel_chat_completions(
      const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle);

  /**
   * @brief Handles an accepted goal for the GenerateChatCompletions action.
   *
   * This method is called when a goal for the GenerateChatCompletions action is
   * accepted.
   *
   * @param goal_handle A shared pointer to the goal handle for the
   * GenerateChatCompletions action.
   */
  void handle_accepted_chat_completions(
      const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle);
};

} // namespace llama_ros

#endif
