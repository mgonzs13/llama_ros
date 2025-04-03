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

#ifndef LLAMA_BT__ACTION__BT_TYPES_HPP_
#define LLAMA_BT__ACTION__BT_TYPES_HPP_

#include <string>
#include <vector>

// Include behavior tree library
#if defined(BTV3)
#include <behaviortree_cpp_v3/behavior_tree.h>
#else
#include "behaviortree_cpp/json_export.h"
#endif

#include "llama_msgs/msg/chat_content.hpp"
#include "llama_msgs/msg/chat_message.hpp"
#include "llama_msgs/msg/chat_req_tool.hpp"
#include "llama_msgs/msg/chat_tool_call.hpp"
#include "llama_msgs/msg/response.hpp"
#include <nlohmann/json.hpp>

#if not defined(BTV3)
// Allow bi-directional convertion to JSON
BT_JSON_CONVERTER(llama_msgs::msg::Response, response) {
  add_field("text", &response.text);
  add_field("tokens", &response.tokens);
}
#endif

// Template specialization
namespace BT {
template <>
inline llama_msgs::msg::Response convertFromString(BT::StringView str) {

#if not defined(BTV3)
  if (StartWith(str, "json:")) {
    str.remove_prefix(5);
    return convertFromJSON<llama_msgs::msg::Response>(str);
  }
#endif

  llama_msgs::msg::Response output;
  if (!str.empty()) {
    // We expect values separated by /
    auto parts = splitString(str, '/');
    if (parts.size() != 2) {
      throw RuntimeError("invalid input)");
    } else {
      output.text = convertFromString<std::string>(parts[0]);
      output.tokens = convertFromString<std::vector<int>>(parts[1]);
    }
  }
  return output;
}

template <>
inline std::vector<std::string> convertFromString(BT::StringView str) {
  auto parts = splitString(str, ';');
  std::vector<std::string> output;
  output.reserve(parts.size());
  for (const StringView &part : parts) {
    output.push_back(convertFromString<std::string>(part));
  }
  return output;
}

// Chat completions parsing
template <>
inline std::vector<llama_msgs::msg::ChatMessage>
convertFromString<std::vector<llama_msgs::msg::ChatMessage>>(
    BT::StringView str) {
  fprintf(stderr, "Parsing chat message: %s\n", str.data());
  std::vector<llama_msgs::msg::ChatMessage> output;
  if (!str.empty()) {
    auto data = nlohmann::json::parse(str.data());

    for (size_t i = 0; i < data.size(); i++) {
      auto message = data[i];
      llama_msgs::msg::ChatMessage chat_message;
      chat_message.role = data["role"];
      if (data.contains("content")) {
        chat_message.content = data["content"];
      }
      if (data.contains("reasoning_content")) {
        chat_message.reasoning_content = data["reasoning_content"];
      }
      if (data.contains("tool_name")) {
        chat_message.tool_name = data["tool_name"];
      }
      if (data.contains("tool_call_id")) {
        chat_message.tool_call_id = data["tool_call_id"];
      }
      if (data.contains("content_parts")) {
        std::vector<llama_msgs::msg::ChatContent> content_parts;
        for (const auto &part : data["content_parts"]) {
          llama_msgs::msg::ChatContent content_part;
          content_part.type = part["type"];
          content_part.text = part["text"];
          content_parts.push_back(content_part);
        }
        chat_message.content_parts = content_parts;
      }
      if (data.contains("tool_calls")) {
        std::vector<llama_msgs::msg::ChatToolCall> tool_calls;
        for (const auto &tool_call : data["tool_calls"]) {
          llama_msgs::msg::ChatToolCall tool_call_msg;
          tool_call_msg.name = tool_call["name"];
          tool_call_msg.id = tool_call["id"];
          tool_call_msg.arguments = tool_call["arguments"];
          tool_calls.push_back(tool_call_msg);
        }
        chat_message.tool_calls = tool_calls;
      }
      output.push_back(chat_message);
    }
  }
  return output;
}

template <>
inline std::vector<llama_msgs::msg::ChatReqTool>
convertFromString<std::vector<llama_msgs::msg::ChatReqTool>>(
    BT::StringView str) {
  std::vector<llama_msgs::msg::ChatReqTool> output;
  if (!str.empty()) {
    auto data = nlohmann::json::parse(str.data());

    for (size_t i = 0; i < data.size(); i++) {
      auto tool = data[i];
      llama_msgs::msg::ChatReqTool chat_req_tool;

      if (data.contains("function")) {
        llama_msgs::msg::ChatTool function;
        function.name = data["function"]["name"];
        function.description = data["function"]["description"];
        function.parameters = data["function"]["parameters"];
        chat_req_tool.function = function;
      }

      output.push_back(chat_req_tool);
    }
  }
  return output;
}

} // namespace BT

#endif // LLAMA_BT__ACTION__BT_TYPES_HPP_