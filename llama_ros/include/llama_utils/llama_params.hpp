// MIT License

// Copyright (c) 2024  Miguel Ángel González Santamarta

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

#ifndef LLAMA_ROS__GPT_PARAMS_HPP
#define LLAMA_ROS__GPT_PARAMS_HPP

#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "common.h"
#include "llama.h"
#include "llama_msgs/msg/sampling_config.hpp"
#include "llava_ros/llava.hpp"

namespace llama_utils {

struct llama_params {
  bool debug;
  struct gpt_params params;
  struct llava_ros::llava_params llava_params;
};

void declare_llama_params(
    const rclcpp_lifecycle::LifecycleNode::SharedPtr &node);

struct llama_params
get_llama_params(const rclcpp_lifecycle::LifecycleNode::SharedPtr &node);

struct llama_sampling_params
parse_sampling_params(const llama_msgs::msg::SamplingConfig &sampling_config,
                      int n_vocab, llama_token token_eos);

} // namespace llama_utils

#endif
