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

#include "llama_msgs/action/generate_chat_completions.hpp"
#include "llama_msgs/action/generate_response.hpp"
#include "llama_ros/llama_node.hpp"
#include "llava_ros/llava.hpp"

namespace llava_ros {

/**
 * @brief Represents a ROS 2 node for managing llava.cpp operations.
 *
 * This class extends the LlamaNode to provide additional functionality for
 * handling Llava-specific operations, such as image processing and chat
 * completions.
 */
class LlavaNode : public llama_ros::LlamaNode {

  /**
   * @brief Action definition for generating responses.
   *
   * This action allows clients to request text responses from the Llava model.
   */
  using GenerateResponse = llama_msgs::action::GenerateResponse;

  /**
   * @brief Goal handle for the GenerateResponse action.
   *
   * This type is used to manage the lifecycle of a goal for the
   * GenerateResponse action.
   */
  using GoalHandleGenerateResponse =
      rclcpp_action::ServerGoalHandle<GenerateResponse>;

  /**
   * @brief Action definition for generating chat completions.
   *
   * This action allows clients to request chat completions from the Llava
   * model.
   */
  using GenerateChatCompletions = llama_msgs::action::GenerateChatCompletions;

  /**
   * @brief Goal handle for the GenerateChatCompletions action.
   *
   * This type is used to manage the lifecycle of a goal for the
   * GenerateChatCompletions action.
   */
  using GoalHandleGenerateChatCompletions =
      rclcpp_action::ServerGoalHandle<GenerateChatCompletions>;

public:
  /**
   * @brief Constructs a new LlavaNode instance.
   *
   * Initializes the node and sets up the necessary services and actions for
   * Llava operations.
   */
  LlavaNode();

protected:
  /**
   * @brief Creates and initializes the Llava instance.
   *
   * This method overrides the base LlamaNode class to create and configure the
   * Llava model.
   */
  void create_llama() override;

  /**
   * @brief Checks if the GenerateResponse goal is empty.
   *
   * This method validates whether the provided goal for the GenerateResponse
   * action is empty.
   *
   * @param goal A shared pointer to the GenerateResponse goal.
   * @return True if the goal is empty, false otherwise.
   */
  bool goal_empty(std::shared_ptr<const GenerateResponse::Goal> goal) override;

  /**
   * @brief Executes the GenerateResponse action.
   *
   * This method handles the execution of the GenerateResponse action for the
   * provided goal handle.
   *
   * @param goal_handle A shared pointer to the goal handle for the
   * GenerateResponse action.
   */
  void execute(
      const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) override;

  /**
   * @brief Checks if the GenerateChatCompletions goal is empty.
   *
   * This method validates whether the provided goal for the
   * GenerateChatCompletions action is empty.
   *
   * @param goal A shared pointer to the GenerateChatCompletions goal.
   * @return True if the goal is empty, false otherwise.
   */
  bool goal_empty_chat_completions(
      std::shared_ptr<const GenerateChatCompletions::Goal> goal) override;

  /**
   * @brief Executes the GenerateChatCompletions action.
   *
   * This method handles the execution of the GenerateChatCompletions action for
   * the provided goal handle.
   *
   * @param goal_handle A shared pointer to the goal handle for the
   * GenerateChatCompletions action.
   */
  void execute_chat_completions(
      const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle)
      override;

  /**
   * @brief Load images from a vector of sensor_msgs::msg::Image messages.
   *
   * This method processes the input images and prepares them.
   */
  bool load_images(std::vector<sensor_msgs::msg::Image> images_msg);

  /**
   * @brief Load audios from a vector of messages.
   *
   * This method processes the input audios and prepares them.
   */
  bool load_audios(std::vector<std_msgs::msg::UInt8MultiArray> audios_msg);
};

} // namespace llava_ros

#endif
