// MIT License
//
// Copyright (c) 2024 Miguel Ángel González Santamarta
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

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "llama_utils/chat_utils.hpp"
#include "llama_msgs/msg/chat_message.h"
#include "llama_msgs/msg/choice.h"
#include "llama_msgs/msg/tool_call.h"

common_chat_templates_inputs llama_utils::parse_chat_completions_goal(
    const std::shared_ptr<
        const llama_msgs::action::GenerateChatCompletions::Goal>
        goal) {
  struct common_chat_templates_inputs inputs;
  std::vector<common_chat_msg> messages;
  for (auto message : goal->messages) {
    struct common_chat_msg msg;
    msg.role = message.role;
    msg.content = message.content;
    std::vector<common_chat_msg_content_part> content_parts;
    for (auto content_part : message.content_parts) {
      struct common_chat_msg_content_part part;
      part.type = content_part.type;
      part.text = content_part.text;
      content_parts.push_back(part);
    }
    msg.content_parts = content_parts;

    std::vector<common_chat_tool_call> tool_calls;
    for (auto tool_call : message.tool_calls) {
      struct common_chat_tool_call call;
      call.name = tool_call.name;
      call.arguments = tool_call.arguments;
      call.id = tool_call.id;
      tool_calls.push_back(call);
    }
    msg.tool_calls = tool_calls;
    messages.push_back(msg);
  }

  inputs.messages = messages;
  inputs.grammar = goal->grammar;
  inputs.json_schema = goal->json_schema;
  inputs.add_generation_prompt = goal->add_generation_prompt;
  inputs.use_jinja = goal->use_jinja;

  return inputs;
}

llama_msgs::action::GenerateChatCompletions::Result
llama_utils::generate_chat_completions_result(const ResponseResult &result) {
  std::string finish_reason = "length";
  common_chat_msg msg;

  // TODO: Implement this
  // if (result.stop == STOP_TYPE_WORD || result.stop == STOP_TYPE_EOS) {
  //   msg = common_chat_parse(result.content, result.oaicompat_chat_format);
  //   finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
  // } else {
    msg.content = result.content;
  // }

  llama_msgs::msg::ChatMessage chat_msg;

  if (!msg.reasoning_content.empty()) {
    chat_msg.reasoning_content = msg.reasoning_content;
  }
  if (!msg.content.empty() || msg.tool_calls.empty()) {
    chat_msg.content = msg.content;
  }
  if (!msg.tool_calls.empty()) {
    std::vector<llama_msgs::msg::ToolCall> tool_calls;
    for (const auto &tc : msg.tool_calls) {
      llama_msgs::msg::ToolCall tool_call;
      tool_call.name = tc.name;
      tool_call.arguments = tc.arguments;
      tool_call.id = tc.id;
      tool_calls.push_back(tool_call);
    }
    chat_msg.tool_calls = tool_calls;
  }

  llama_msgs::msg::Choice choice;
  choice.finish_reason = finish_reason;
  choice.index = 0;
  choice.message = chat_msg;

  // TODO: Logprobs
  // if (!stream && probs_output.size() > 0) {
  //     choice["logprobs"] = json{
  //         {"content",
  //         completion_token_output::probs_vector_to_json(probs_output,
  //         post_sampling_probs)},
  //     };
  // }

  std::time_t t = std::time(0);

  llama_msgs::action::GenerateChatCompletions::Result res;
  res.choices.push_back(choice);
  res.created = t;
  res.model = result.oaicompat_model;
  res.system_fingerprint = result.build_info;
  res.object = "chat.completion";
  res.usage.completion_tokens = result.n_decoded;
  res.usage.prompt_tokens = result.n_prompt_tokens;
  res.usage.total_tokens = result.n_decoded + result.n_prompt_tokens;
  res.id = result.oaicompat_cmpl_id;

  return res;
}
