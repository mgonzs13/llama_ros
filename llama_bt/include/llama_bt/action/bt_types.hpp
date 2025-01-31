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

#ifndef LLAMA_BT__ACTION__BT_TYPES_HPP_
#define LLAMA_BT__ACTION__BT_TYPES_HPP_

#include <behaviortree_cpp_v3/behavior_tree.h>

#include <string>
#include <vector>

#include "llama_msgs/msg/response.hpp"

// Template specialization
namespace BT
{
template<> inline llama_msgs::msg::Response convertFromString(BT::StringView str)
{
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
}  // namespace BT

#endif  // LLAMA_BT__ACTION__BT_TYPES_HPP_
