// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
// Copyright (c) 2026 Alejandro González Cantón
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

#ifndef LLAVA_ROS__LLAVA_REQUEST_HANDLER_HPP
#define LLAVA_ROS__LLAVA_REQUEST_HANDLER_HPP

#include "llama_ros/request_handler.hpp"
#include "mtmd.h"

namespace llava_ros {

// Forward declaration
class Llava;

/**
 * @brief Handles text completion requests with multimodal support.
 *
 * This handler extends the base RequestHandler to process completion
 * requests that may include image or audio data alongside the text prompt.
 */
class LlavaCompletionRequestHandler : public llama_ros::RequestHandler {
public:
  /**
   * @brief Constructs a LlavaCompletionRequestHandler.
   *
   * @param llava Pointer to the Llava instance that owns this handler.
   */
  explicit LlavaCompletionRequestHandler(Llava *llava);

  /**
   * @brief Prepares a slot for text completion with multimodal support.
   *
   * @param input_prompt The input prompt.
   * @param slot The slot to prepare.
   * @param sparams Sampling parameters.
   * @param callback Callback for streaming results.
   * @param stop Stop sequences.
   * @param reset Whether to reset the slot.
   */
  void handle(
      const std::string &input_prompt, llama_ros::ServerSlot *slot,
      common_params_sampling sparams,
      std::function<void(llama_ros::CompletionOutput, llama_ros::ServerSlot *)>
          callback,
      std::vector<std::string> stop, bool reset);

private:
  Llava *llava_;
};

/**
 * @brief Handles chat completion requests with multimodal support.
 *
 * This handler extends the base RequestHandler to process chat completion
 * requests that may include image or audio data alongside chat messages.
 */
class LlavaChatCompletionRequestHandler : public llama_ros::RequestHandler {
public:
  /**
   * @brief Constructs a LlavaChatCompletionRequestHandler.
   *
   * @param llava Pointer to the Llava instance that owns this handler.
   */
  explicit LlavaChatCompletionRequestHandler(Llava *llava);

  /**
   * @brief Prepares a slot for chat completion with multimodal support.
   *
   * @param chat_context The chat context containing messages and parameters.
   * @param slot The slot to prepare.
   * @param callback Callback for streaming results.
   */
  void handle(
      llama_utils::ChatCompletionsContext chat_context,
      llama_ros::ServerSlot *slot,
      std::function<void(llama_ros::CompletionOutput, llama_ros::ServerSlot *)>
          callback);

private:
  Llava *llava_;
};

} // namespace llava_ros

#endif
