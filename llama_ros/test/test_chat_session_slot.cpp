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

#include <gtest/gtest.h>

#include <vector>

#include "llama_ros/server_slot.hpp"

namespace {

// A slot with a default (template-less) session parses generated text as raw
// content; this is the path used by plain text completions.
TEST(ChatSessionSlot, DefaultSessionIsContentOnly) {
  llama_ros::ServerSlot slot;
  slot.generated_text = "hello world";
  slot.stop = llama_ros::StopType::NO_STOP;
  std::vector<common_chat_msg_diff> diffs;

  const auto &msg = slot.update_chat_msg(diffs);
  EXPECT_EQ(msg.content, "hello world");
}

// Streaming feeds deltas into the session; the final call finishes it once and
// repeated final calls must be safe no-ops.
TEST(ChatSessionSlot, FinishRunsOnce) {
  llama_ros::ServerSlot slot;
  std::vector<common_chat_msg_diff> diffs;

  slot.generated_text = "a";
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  slot.generated_text = "ab";
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  slot.stop = llama_ros::StopType::FULL_STOP;
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  // must not assert/throw on repeated final calls
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  EXPECT_EQ(slot.chat_msg.content, "ab");
}

// reset() restores a fresh session and re-enables feeding.
TEST(ChatSessionSlot, ResetRestoresDefaultSession) {
  llama_ros::ServerSlot slot;
  std::vector<common_chat_msg_diff> diffs;

  slot.generated_text = "x";
  slot.stop = llama_ros::StopType::FULL_STOP;
  slot.update_chat_msg(diffs);
  EXPECT_TRUE(slot.chat_finished);

  slot.reset();
  EXPECT_FALSE(slot.chat_finished);
  EXPECT_EQ(slot.chat_fed_len, 0u);

  slot.generated_text = "y";
  slot.stop = llama_ros::StopType::NO_STOP;
  EXPECT_NO_THROW(slot.update_chat_msg(diffs));
  EXPECT_EQ(slot.chat_msg.content, "y");
}

} // namespace
