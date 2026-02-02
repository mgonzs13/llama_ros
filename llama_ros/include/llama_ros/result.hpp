// MIT License
//
// Copyright (c) 2023 Miguel Ángel González Santamarta
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

#ifndef LLAMA_ROS__RESULT_HPP
#define LLAMA_ROS__RESULT_HPP

#include <optional>
#include <string>
#include <stdexcept>

namespace llama_ros {

/**
 * @brief A Result type for explicit error handling.
 *
 * This type represents either a successful value or an error message.
 * It forces explicit error handling rather than relying on exceptions
 * or silent failures.
 *
 * @tparam T The type of the success value.
 */
template<typename T>
class Result {
public:
  /**
   * @brief Creates a successful result.
   *
   * @param value The success value.
   * @return A Result containing the value.
   */
  static Result ok(T value) {
    Result r;
    r.value_ = std::move(value);
    return r;
  }

  /**
   * @brief Creates an error result.
   *
   * @param error The error message.
   * @return A Result containing the error.
   */
  static Result error(std::string error) {
    Result r;
    r.error_ = std::move(error);
    return r;
  }

  /**
   * @brief Checks if the result is successful.
   *
   * @return True if the result contains a value, false otherwise.
   */
  bool is_ok() const {
    return value_.has_value();
  }

  /**
   * @brief Checks if the result is an error.
   *
   * @return True if the result contains an error, false otherwise.
   */
  bool is_error() const {
    return error_.has_value();
  }

  /**
   * @brief Gets the success value.
   *
   * @throws std::runtime_error if the result is an error.
   * @return The success value.
   */
  T& value() {
    if (!value_.has_value()) {
      throw std::runtime_error("Attempted to access value of error Result: " + 
                               error_.value_or("unknown error"));
    }
    return *value_;
  }

  /**
   * @brief Gets the success value (const version).
   *
   * @throws std::runtime_error if the result is an error.
   * @return The success value.
   */
  const T& value() const {
    if (!value_.has_value()) {
      throw std::runtime_error("Attempted to access value of error Result: " + 
                               error_.value_or("unknown error"));
    }
    return *value_;
  }

  /**
   * @brief Gets the error message.
   *
   * @return The error message, or empty string if successful.
   */
  std::string error() const {
    return error_.value_or("");
  }

  /**
   * @brief Gets the value or a default.
   *
   * @param default_value The default value to return if error.
   * @return The value if successful, otherwise the default.
   */
  T value_or(T default_value) const {
    return value_.value_or(std::move(default_value));
  }

private:
  Result() = default;
  
  std::optional<T> value_;
  std::optional<std::string> error_;
};

} // namespace llama_ros

#endif // LLAMA_ROS__RESULT_HPP
