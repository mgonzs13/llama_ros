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

#include <memory>
#include <string>

#include "llama_bt/action/generate_response_action.hpp"

namespace llama_bt {

GenerateResponseAction::GenerateResponseAction(
    const std::string &xml_tag_name, const std::string &action_name,
    const BT::NodeConfiguration &conf)
    : nav2_behavior_tree::BtActionNode<llama_msgs::action::GenerateResponse>(
          xml_tag_name, action_name, conf) {}

void GenerateResponseAction::on_tick() {
  std::string prompt;
  getInput("prompt", prompt);
  std::vector<std::string> stop;
  getInput("stop", stop);
  bool reset;
  getInput("reset", reset);

  goal_.prompt = prompt;
  goal_.stop = stop;
  goal_.reset = reset;
}

BT::NodeStatus GenerateResponseAction::on_success() {
  setOutput("response", result_.result->response.text);
  return BT::NodeStatus::SUCCESS;
}

} // namespace llama_bt

#include "behaviortree_cpp/bt_factory.h"
BT_REGISTER_NODES(factory) {
  BT::NodeBuilder builder = [](const std::string &name,
                               const BT::NodeConfiguration &config) {
    return std::make_unique<llama_bt::GenerateResponseAction>(
        name, "generate_response", config);
  };

  factory.registerBuilder<llama_bt::GenerateResponseAction>("GenerateResponse",
                                                            builder);
}
