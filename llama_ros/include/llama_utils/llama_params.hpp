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

#ifndef LLAMA_UTILS__LLAMA_PARAMS_HPP
#define LLAMA_UTILS__LLAMA_PARAMS_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "common.h"

#include "llama_msgs/msg/sampling_config.hpp"
#include "llava_ros/llava.hpp"

namespace llama_utils {

/**
 * @brief Represents the parameters for configuring the Llama system.
 */
struct LlamaParams {
  /**
   * @brief The system prompt used for initializing the Llama system.
   */
  std::string system_prompt;

  /**
   * @brief Common parameters for configuring the Llama system.
   */
  struct common_params params;
};

/**
 * @brief Declares the parameters for the Llama system.
 *
 * @param node The lifecycle node to which the parameters will be declared.
 */
void declare_llama_params(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr &node);

/**
 * @brief Retrieves the Llama parameters from the given lifecycle node.
 *
 * @param node The shared pointer to the lifecycle node from which parameters
 * will be retrieved.
 * @return A struct containing the Llama parameters.
 */
struct LlamaParams
get_llama_params(const rclcpp_lifecycle::LifecycleNode::SharedPtr &node);

/**
 * @brief Parses a scheduling priority from a string.
 *
 * @param priority The string representing the scheduling priority.
 * @return The parsed scheduling priority as an enum value.
 */
enum ggml_sched_priority parse_priority(std::string priority);

/**
 * @brief Parses a grammar trigger type from an integer.
 *
 * @param type The integer representing the grammar trigger type.
 * @return The parsed grammar trigger type as an enum value.
 */
common_grammar_trigger_type parse_grammar_trigger_type(int type);

/**
 * @brief Parses sampling parameters from a SamplingConfig message.
 *
 * @param sampling_config The SamplingConfig message containing sampling
 * configuration.
 * @param n_vocab The size of the vocabulary.
 * @return A struct containing the parsed sampling parameters.
 */
struct common_params_sampling
parse_sampling_params(const llama_msgs::msg::SamplingConfig &sampling_config,
                      int n_vocab);

} // namespace llama_utils

#endif
