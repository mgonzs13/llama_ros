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
using std::placeholders::_1;
using std::placeholders::_2;

LlavaNode::LlavaNode() : llama_ros::LlamaNode() {}

void LlavaNode::create_llama() {
  this->llama =
      std::make_unique<Llava>(this->params.params, this->params.llava_params,
                              this->params.system_prompt);
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
    auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
    std::string encoded_image = this->base64_encode(enc_msg, buf.size());

    if (!static_cast<Llava *>(this->llama.get())->load_image(encoded_image)) {
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
    auto *enc_msg = reinterpret_cast<unsigned char *>(buf.data());
    std::string encoded_image = this->base64_encode(enc_msg, buf.size());

    if (!static_cast<Llava *>(this->llama.get())->load_image(encoded_image)) {
      this->goal_handle_chat_->abort(result);
      RCLCPP_ERROR(this->get_logger(), "Failed to load image");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Image loaded");
  }

  // llama_node execute_chat_completions
  llama_ros::LlamaNode::execute_chat_completions(goal_handle);
}

// https://renenyffenegger.ch/notes/development/Base64/Encoding-and-decoding-base-64-with-cpp/
std::string LlavaNode::base64_encode(unsigned char const *bytes_to_encode,
                                     size_t in_len, bool url) {

  static const char *base64_chars[2] = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789"
                                        "+/",

                                        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                        "abcdefghijklmnopqrstuvwxyz"
                                        "0123456789"
                                        "-_"};

  size_t len_encoded = (in_len + 2) / 3 * 4;

  unsigned char trailing_char = url ? '.' : '=';

  //
  // Choose set of base64 characters. They differ
  // for the last two positions, depending on the url
  // parameter.
  // A bool (as is the parameter url) is guaranteed
  // to evaluate to either 0 or 1 in C++ therefore,
  // the correct character set is chosen by subscripting
  // base64_chars with url.
  //
  const char *base64_chars_ = base64_chars[url];

  std::string ret;
  ret.reserve(len_encoded);

  unsigned int pos = 0;

  while (pos < in_len) {
    ret.push_back(base64_chars_[(bytes_to_encode[pos + 0] & 0xfc) >> 2]);

    if (pos + 1 < in_len) {
      ret.push_back(base64_chars_[((bytes_to_encode[pos + 0] & 0x03) << 4) +
                                  ((bytes_to_encode[pos + 1] & 0xf0) >> 4)]);

      if (pos + 2 < in_len) {
        ret.push_back(base64_chars_[((bytes_to_encode[pos + 1] & 0x0f) << 2) +
                                    ((bytes_to_encode[pos + 2] & 0xc0) >> 6)]);
        ret.push_back(base64_chars_[bytes_to_encode[pos + 2] & 0x3f]);
      } else {
        ret.push_back(base64_chars_[(bytes_to_encode[pos + 1] & 0x0f) << 2]);
        ret.push_back(trailing_char);
      }
    } else {

      ret.push_back(base64_chars_[(bytes_to_encode[pos + 0] & 0x03) << 4]);
      ret.push_back(trailing_char);
      ret.push_back(trailing_char);
    }

    pos += 3;
  }

  return ret;
}
