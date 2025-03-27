// MIT License
//
// Copyright (c) 2025 Alberto J. Tudela Roldán
// Copyright (c) 2025 Alejandro González Cantón
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

#ifndef LLAMA_BT__ACTION__GENERATE_CHAT_COMPLETIONS_ACTION_HPP_
#define LLAMA_BT__ACTION__GENERATE_CHAT_COMPLETIONS_ACTION_HPP_

#include <behaviortree_cpp/basic_types.h>
#include <string>
#include <vector>

#include "llama_msgs/action/generate_chat_completions.hpp"
#include "llama_msgs/msg/chat_message.hpp"
#include "llama_msgs/msg/chat_req_tool.hpp"
#include "nav2_behavior_tree/bt_action_node.hpp"

namespace llama_bt {

/**
 * @brief A nav2_behavior_tree::BtActionNode class that wraps
 * llama_msgs::action::GenerateChatCompletions
 */
class GenerateChatCompletionsAction
    : public nav2_behavior_tree::BtActionNode<
          llama_msgs::action::GenerateChatCompletions> {
public:
  /**
   * @brief A constructor for llama_bt::GenerateChatCompletions Service
   * @param xml_tag_name Name for the XML tag for this node
   * @param action_name Action name this node creates a client for
   * @param conf BT node configuration
   */
  GenerateChatCompletionsAction(const std::string &xml_tag_name,
                                const std::string &action_name,
                                const BT::NodeConfiguration &conf);

  /**
   * @brief Function to perform some user-defined operation on tick
   * @return BT::NodeStatus Status of tick execution
   */
  void on_tick() override;

  /**
   * @brief Function to perform some user-defined operation upon successful
   * completion of the action
   */
  BT::NodeStatus on_success() override;

  /**
   * @brief Creates list of BT ports
   * @return BT::PortsList Containing node-specific ports
   */
  static BT::PortsList providedPorts() {
    return providedBasicPorts({
        BT::InputPort<std::vector<llama_msgs::msg::ChatMessage>>(
            "messages", "Chat messages"),
        BT::InputPort<std::vector<llama_msgs::msg::ChatReqTool>>(
            "tools", "Chat request tools"),
        BT::InputPort<std::string>("tool_choice", "auto", "Tool choice"),

        BT::OutputPort<llama_msgs::msg::ChatMessage>("choice_message",
                                                     "Chat choice message"),
    });
  }
};

} // namespace llama_bt

#endif // LLAMA_BT__ACTION__GENERATE_CHAT_COMPLETIONS_ACTION_HPP_