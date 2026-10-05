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

#ifndef LLAMA_ROS__METADATA_UTILS_HPP
#define LLAMA_ROS__METADATA_UTILS_HPP

#include <string>
#include <vector>

namespace llama_ros {

/**
 * @brief Parses llama.cpp's stringified flat GGUF array values.
 *
 * Input looks like [a, b, ...] or ["a", "b", ...] with \\ and \" escapes.
 * Nested arrays, unterminated quotes and other malformed input yield an
 * empty vector; never throws.
 */
std::vector<std::string> parse_metadata_array(const std::string &value);

/**
 * @brief Parses an integer metadata value, returning fallback on failure.
 */
int parse_metadata_int(const std::string &value, int fallback = 0);

/**
 * @brief Parses a float metadata value, returning fallback on failure.
 */
float parse_metadata_float(const std::string &value, float fallback = 0.0f);

/**
 * @brief Splits a ";"-separated metadata list (e.g. sampling.sequence).
 */
std::vector<std::string> split_semicolon(const std::string &value);

} // namespace llama_ros

#endif // LLAMA_ROS__METADATA_UTILS_HPP
