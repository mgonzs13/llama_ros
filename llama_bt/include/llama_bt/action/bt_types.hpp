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

#include <behaviortree_cpp_v3/behavior_tree.h>

#include <string>
#include <vector>

#include "llama_msgs/msg/response.hpp"

// Template specialization
namespace BT {
template <>
inline llama_msgs::msg::Response convertFromString(BT::StringView str) {
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
} // namespace BT

#endif // LLAMA_BT__ACTION__BT_TYPES_HPP_
