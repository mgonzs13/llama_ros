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

#if defined(JAZZY)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif

#include <memory>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "llama_utils/llama_params.hpp"
#include "llava_ros/llava_node.hpp"

using namespace llava_ros;

LlavaNode::LlavaNode() : llama_ros::LlamaNode() {}

void LlavaNode::create_llama() {
  this->llama =
      std::make_unique<Llava>(this->params.params, this->params.system_prompt);
}

bool LlavaNode::goal_empty(std::shared_ptr<const GenerateResponse::Goal> goal) {
  return goal->prompt.size() == 0 && goal->image.data.size() == 0;
}

void LlavaNode::execute(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) {

  auto result = std::make_shared<GenerateResponse::Result>();
  auto image_msg = goal_handle->get_goal()->image;

  // load image
  if (image_msg.data.size() > 0) {

    RCLCPP_INFO(this->get_logger(), "Loading image...");

    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(image_msg, image_msg.encoding);

    std::vector<uchar> buf;
    cv::imencode(".jpg", cv_ptr->image, buf);

    if (!static_cast<Llava *>(this->llama.get())->load_image(buf)) {
      this->goal_handle_->abort(result);
      RCLCPP_ERROR(this->get_logger(), "Failed to load image");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Image loaded");
  }

  // llama_node execute
  llama_ros::LlamaNode::execute(goal_handle);
}

/*
************************
*    CHAT COMPLETIONS  *
************************
*/
bool LlavaNode::goal_empty_chat_completions(
    std::shared_ptr<const GenerateChatCompletions::Goal> goal) {
  return goal->messages.size() == 0 && goal->image.data.size() == 0;
}

void LlavaNode::execute_chat_completions(
    const std::shared_ptr<GoalHandleGenerateChatCompletions> goal_handle) {

  auto result = std::make_shared<GenerateChatCompletions::Result>();
  auto image_msg = goal_handle->get_goal()->image;

  RCLCPP_INFO(this->get_logger(), "Executing chat completions");

  // load image
  if (image_msg.data.size() > 0) {

    RCLCPP_INFO(this->get_logger(), "Loading image...");

    cv_bridge::CvImagePtr cv_ptr =
        cv_bridge::toCvCopy(image_msg, image_msg.encoding);

    std::vector<uchar> buf;
    cv::imencode(".jpg", cv_ptr->image, buf);

    if (!static_cast<Llava *>(this->llama.get())->load_image(buf)) {
      this->goal_handle_chat_->abort(result);
      RCLCPP_ERROR(this->get_logger(), "Failed to load image");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Image loaded");
  }

  // llama_node execute_chat_completions
  llama_ros::LlamaNode::execute_chat_completions(goal_handle);
}
