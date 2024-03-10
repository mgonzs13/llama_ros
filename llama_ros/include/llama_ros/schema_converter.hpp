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

#ifndef LLAMA_ROS__SCHEMA_CONVERTER_HPP
#define LLAMA_ROS__SCHEMA_CONVERTER_HPP

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace llama_ros {

const std::string SPACE_RULE = "\" \"?";

const std::map<std::string, std::string> PRIMITIVE_RULES = {
    {"boolean", "(\"true\" | \"false\") space"},
    {"number", "(\"-\"? ([0-9] | [1-9] [0-9]*)) (\".\" [0-9]+)? ([eE] [-+]? "
               "[0-9]+)? space"},
    {"integer", "(\"-\"? ([0-9] | [1-9] [0-9]*)) space"},
    {"string",
     R"("\"" ([^"\\\x7F\x00-\x1F] |"\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]) )* "\"" space)"},
    {"null", "\"null\" space"}};

class SchemaConverter {
public:
  SchemaConverter(const std::vector<std::string> &prop_order);
  std::string visit(const json &schema, const std::string &name = "");
  std::string format_grammar();

  static std::string
  json_schema_to_gbnf(const std::string &schema_str,
                      const std::vector<std::string> &prop_order) {

    json schema = json::parse(schema_str);
    SchemaConverter converter(prop_order);
    converter.visit(schema, "");

    return converter.format_grammar();
  }

private:
  std::map<std::string, std::string> rules;
  std::map<std::string, int> prop_order;

  std::string format_literal(const json &literal);
  std::string add_rule(const std::string &name, const std::string &rule);
};

} // namespace llama_ros

#endif
