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

#include "llama_utils/chat_formatter.hpp"
#include "llama_utils/logs.hpp"

namespace llama_utils {

ChatFormatter::ChatFormatter(const llama_model* model, const std::string& custom_template)
    : custom_template_(custom_template) {
  
  chat_templates_ = chat_templates_ptr(
    common_chat_templates_init(model, custom_template.empty() ? "" : custom_template.c_str())
  );
  
  if (!chat_templates_) {
    LLAMA_LOG_WARN("Failed to initialize chat templates");
  }
  LLAMA_LOG_INFO("Initialized Chat Formatter");
}

ChatFormatter::~ChatFormatter() {
  // Unique_ptr handles deletion
}

common_chat_params ChatFormatter::apply_template(const common_chat_templates_inputs& inputs) {
  if (!chat_templates_) {
    LLAMA_LOG_ERROR("Chat templates not initialized");
    return common_chat_params{};
  }

  return common_chat_templates_apply(chat_templates_.get(), inputs);
}

common_chat_msg ChatFormatter::parse_response(
    const std::string& text,
    bool is_partial,
    const common_chat_syntax& chat_syntax) {
  
  return common_chat_parse(text, is_partial, chat_syntax);
}

} // namespace llama_utils
