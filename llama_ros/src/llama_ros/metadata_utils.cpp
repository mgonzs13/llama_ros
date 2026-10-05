// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
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

#include "llama_ros/metadata_utils.hpp"

#include <charconv>
#include <system_error>

namespace llama_ros {

namespace {

std::string trim(const std::string &value) {
  const char *spaces = " \t\n\r";
  const size_t begin = value.find_first_not_of(spaces);

  if (begin == std::string::npos) {
    return "";
  }

  const size_t end = value.find_last_not_of(spaces);
  return value.substr(begin, end - begin + 1);
}

} // namespace

std::vector<std::string> parse_metadata_array(const std::string &value) {
  std::vector<std::string> out;

  const std::string text = trim(value);
  if (text.size() < 2 || text.front() != '[' || text.back() != ']') {
    return out;
  }

  std::string current;
  bool in_quotes = false;
  bool escaped = false;
  bool element_quoted = false;

  auto push_current = [&out, &current, &element_quoted]() {
    // spaces inside quotes are meaningful, outside they are separators
    const std::string item = element_quoted ? current : trim(current);
    if (!item.empty() || element_quoted) {
      out.push_back(item);
    }
    current.clear();
    element_quoted = false;
  };

  for (size_t i = 1; i + 1 < text.size(); ++i) {
    const char c = text[i];

    if (escaped) {
      current.push_back(c);
      escaped = false;
      continue;
    }

    if (c == '\\') {
      escaped = true;
      continue;
    } else if (c == '"') {
      if (!in_quotes && !element_quoted) {
        // separator whitespace before the opening quote is not content
        current = trim(current);
      }
      in_quotes = !in_quotes;
      element_quoted = true;
      continue;
    }

    if (!in_quotes && element_quoted &&
        (c == ' ' || c == '\t' || c == '\n' || c == '\r')) {
      // separator whitespace after the closing quote is not content
      continue;
    }

    if (!in_quotes && (c == '[' || c == ']')) {
      // nested arrays are not produced by llama.cpp's stringification
      return {};
    }

    if (c == ',' && !in_quotes) {
      push_current();
      continue;
    }

    current.push_back(c);
  }

  if (in_quotes || escaped) {
    return {};
  }

  push_current();

  return out;
}

int parse_metadata_int(const std::string &value, int fallback) {
  const std::string text = trim(value);

  if (text.empty()) {
    return fallback;
  }

  int result = 0;
  const auto [ptr, ec] =
      std::from_chars(text.data(), text.data() + text.size(), result);

  if (ec != std::errc() || ptr != text.data() + text.size()) {
    return fallback;
  }

  return result;
}

float parse_metadata_float(const std::string &value, float fallback) {
  const std::string text = trim(value);

  if (text.empty()) {
    return fallback;
  }

  float result = 0.0f;
  const auto [ptr, ec] =
      std::from_chars(text.data(), text.data() + text.size(), result);

  if (ec != std::errc() || ptr != text.data() + text.size()) {
    return fallback;
  }

  return result;
}

std::vector<std::string> split_semicolon(const std::string &value) {
  std::vector<std::string> out;
  std::string current;

  for (const char c : value) {
    if (c == ';') {
      const std::string item = trim(current);

      if (!item.empty()) {
        out.push_back(item);
      }

      current.clear();
      continue;
    }

    current.push_back(c);
  }

  const std::string item = trim(current);

  if (!item.empty()) {
    out.push_back(item);
  }

  return out;
}

} // namespace llama_ros
