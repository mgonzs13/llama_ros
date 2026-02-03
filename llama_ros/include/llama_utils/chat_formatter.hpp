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

#ifndef LLAMA_UTILS__CHAT_FORMATTER_HPP
#define LLAMA_UTILS__CHAT_FORMATTER_HPP

#include <memory>
#include <string>
#include <map>

#include "chat.h"
#include "common.h"
#include "llama.h"

namespace llama_utils {

/**
 * @brief Handles chat template formatting and parsing.
 *
 * This utility encapsulates all chat template logic, keeping it separate
 * from the core Llama model management. It can format messages using
 * various chat templates (ChatML, Llama-2, etc.) and parse responses.
 */
class ChatFormatter {
public:
  /**
   * @brief Constructor for ChatFormatter.
   *
   * @param model The llama model to extract templates from.
   * @param custom_template Optional custom template override.
   */
  ChatFormatter(const llama_model* model, const std::string& custom_template = "");

  /**
   * @brief Destructor.
   */
  ~ChatFormatter();

  /**
   * @brief Applies a chat template to format messages.
   *
   * @param inputs The chat template inputs (messages, tools, etc.).
   * @return The formatted prompt string and metadata.
   */
  common_chat_params apply_template(const common_chat_templates_inputs& inputs);

  /**
   * @brief Gets the raw chat templates object.
   *
   * @return Pointer to the chat templates.
   */
  common_chat_templates* get_templates() const {
    return chat_templates_.get();
  }

  /**
   * @brief Parses a response string into structured chat message.
   *
   * @param text The raw response text.
   * @param is_partial Whether this is a partial response (streaming).
   * @param chat_syntax The chat syntax being used.
   * @return The parsed chat message.
   */
  static common_chat_msg parse_response(
    const std::string& text,
    bool is_partial,
    const common_chat_parser_params& chat_syntax
  );

private:
  using chat_templates_ptr = std::unique_ptr<
    common_chat_templates,
    common_chat_templates_deleter
  >;

  chat_templates_ptr chat_templates_;
  std::string custom_template_;
};

} // namespace llama_utils

#endif // LLAMA_UTILS__CHAT_FORMATTER_HPP
