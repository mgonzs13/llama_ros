// MIT License
//
// Copyright (c) 2025 Alberto J. Tudela Roldán
// Copyright (c) 2025 Grupo Avispa, DTE, Universidad de Málaga
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
