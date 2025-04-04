// MIT License
//
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

#include <memory>
#include <string>

#include "llama_bt/action/generate_chat_completions_action.hpp"
#include "llama_msgs/msg/chat_tool.hpp"

namespace llama_bt {

GenerateChatCompletionsAction::GenerateChatCompletionsAction(
    const std::string &xml_tag_name, const std::string &action_name,
    const BT::NodeConfiguration &conf)
    : llama_bt::BtActionNode<llama_msgs::action::GenerateChatCompletions>(
          xml_tag_name, action_name, conf) {}

void GenerateChatCompletionsAction::on_tick() {
  std::vector<llama_msgs::msg::ChatMessage> chat_messages;
  getInput("messages", chat_messages);
  std::vector<llama_msgs::msg::ChatReqTool> chat_req_tools;
  getInput("tools", chat_req_tools);
  std::string tool_choice;
  getInput("tool_choice", tool_choice);

  goal_.messages = chat_messages;
  goal_.tools = chat_req_tools;

  if (tool_choice == "required") {
    goal_.tool_choice = llama_msgs::msg::ChatTool::TOOL_CHOICE_REQUIRED;
  } else if (tool_choice == "none") {
    goal_.tool_choice = llama_msgs::msg::ChatTool::TOOL_CHOICE_NONE;
  } else {
    goal_.tool_choice = llama_msgs::msg::ChatTool::TOOL_CHOICE_AUTO;
  }

  goal_.add_generation_prompt = true;
  goal_.use_jinja = true;
  goal_.parallel_tool_calls = chat_req_tools.size() > 1;
  goal_.stream = false;
}

BT::NodeStatus GenerateChatCompletionsAction::on_success() {
  setOutput("choice_message", result_.result->choices[0].message);
  return BT::NodeStatus::SUCCESS;
}

} // namespace llama_bt

#if defined(BTV3)
#include "behaviortree_cpp_v3/bt_factory.h"
#else
#include "behaviortree_cpp/bt_factory.h"
#endif

BT_REGISTER_NODES(factory) {
  BT::NodeBuilder builder = [](const std::string &name,
                               const BT::NodeConfiguration &config) {
    return std::make_unique<llama_bt::GenerateChatCompletionsAction>(
        name, "generate_chat_completions", config);
  };

  factory.registerBuilder<llama_bt::GenerateChatCompletionsAction>(
      "GenerateChatCompletions", builder);
}