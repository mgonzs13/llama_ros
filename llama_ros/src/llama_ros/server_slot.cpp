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

#include "llama_ros/server_slot.hpp"
#include "llama_utils/chat_utils.hpp"
#include "llama_utils/logs.hpp"

using namespace llama_ros;

const common_chat_msg &ServerSlot::update_chat_msg(
    std::vector<common_chat_msg_diff> &oaicompat_msg_diffs) {

  auto previous_msg = this->chat_msg;
  auto new_msg =
      common_chat_parse(this->generated_text,
                        /* is_partial= */ this->stop != StopType::FULL_STOP,
                        this->params.oaicompat_chat_syntax);

  if (!new_msg.empty()) {
    std::function<std::string()> gen_tool_call_id =
        static_cast<std::string (*)()>(llama_utils::random_string);
    new_msg.set_tool_call_ids(this->generated_tool_call_ids, gen_tool_call_id);
    this->chat_msg = new_msg;
    oaicompat_msg_diffs = common_chat_msg_diff::compute_diffs(
        previous_msg, new_msg.empty() ? previous_msg : new_msg);
  }

  return this->chat_msg;
}

void ServerSlot::reset() {
  this->generated_text.clear();
  this->stop = StopType::NO_STOP;
  this->stopping_word.clear();
  this->n_past = 0;
  this->n_sent_text = 0;
  this->chat_format = COMMON_CHAT_FORMAT_CONTENT_ONLY;
  this->generated_tool_call_ids.clear();
  this->chat_msg = {};

  this->generated_probs.clear();

  this->map_pos_to_media.clear();
  this->prompt_tokens.clear();
}

void ServerSlot::release() {
  LLAMA_LOG_INFO("Trying to release slot %d", this->id);
  if (this->is_processing()) {
    LLAMA_LOG_INFO("Releasing slot %d", this->id);
    this->reset();
    this->state = SLOT_STATE_IDLE;
  }
}

size_t ServerSlot::find_stopping_strings(const std::string &text,
                                         const size_t last_token_size,
                                         bool is_full_stop) {
  size_t stop_pos = std::string::npos;

  for (const std::string &word : this->params.antiprompt) {
    size_t pos;

    if (is_full_stop) {
      const size_t tmp = word.size() + last_token_size;
      const size_t from_pos = text.size() > tmp ? text.size() - tmp : 0;

      pos = text.find(word, from_pos);
    } else {
      pos = string_find_partial_stop(text, word);
    }

    if (pos != std::string::npos &&
        (stop_pos == std::string::npos || pos < stop_pos)) {
      if (is_full_stop) {
        this->stop = FULL_STOP;
        this->stopping_word = word;
        this->has_next_token = false;
      }
      stop_pos = pos;
    }
  }

  return stop_pos;
}
