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
#include "behaviortree_cpp/behavior_tree.h"
#include "behaviortree_cpp/json_export.h"
#endif

#include "llama_msgs/msg/response.hpp"

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
} // namespace BT

#endif // LLAMA_BT__ACTION__BT_TYPES_HPP_