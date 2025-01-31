// Copyright (c) 2025 Alberto J. Tudela Roldán
// Copyright (c) 2025 Grupo Avispa, DTE, Universidad de Málaga
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LLAMA_BT__ACTION__GENERATE_RESPONSE_ACTION_HPP_
#define LLAMA_BT__ACTION__GENERATE_RESPONSE_ACTION_HPP_

#include <memory>
#include <string>
#include <vector>

#include "llama_bt/action/bt_types.hpp"
#include "llama_msgs/action/generate_response.hpp"
#include "llama_msgs/msg/response.hpp"
#include "nav2_behavior_tree/bt_action_node.hpp"


namespace llama_bt
{

/**
 * @brief A nav2_behavior_tree::BtActionNode class that wraps llama_msgs::action::GenerateResponse
 */
class GenerateResponseAction
  : public nav2_behavior_tree::BtActionNode<llama_msgs::action::GenerateResponse>
{
public:
  /**
   * @brief A constructor for llama_bt::GenerateResponse Service
   * @param xml_tag_name Name for the XML tag for this node
   * @param action_name Action name this node creates a client for
   * @param conf BT node configuration
   */
  GenerateResponseAction(
    const std::string & xml_tag_name,
    const std::string & action_name,
    const BT::NodeConfiguration & conf);

  /**
   * @brief Function to perform some user-defined operation on tick
   * @return BT::NodeStatus Status of tick execution
   */
  void on_tick() override;

  /**
   * @brief Function to perform some user-defined operation upon successful completion of the action
   */
  BT::NodeStatus on_success() override;

  /**
   * @brief Creates list of BT ports
   * @return BT::PortsList Containing node-specific ports
   */
  static BT::PortsList providedPorts()
  {
    return providedBasicPorts(
      {
        BT::InputPort<std::string>("prompt", "Prompt"),
        BT::InputPort<std::vector<std::string>>("stop", "Stop list"),
        BT::InputPort<bool>("reset", false, "Whether to reset the context"),
        BT::OutputPort<std::string>("response", "Final Response"),
      });
  }
};

}  // namespace llama_bt

#endif  // LLAMA_BT__ACTION__GENERATE_RESPONSE_ACTION_HPP_
