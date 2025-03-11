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
#include "llama_msgs/msg/chat_choice.h"
#include "llama_msgs/msg/chat_message.h"
#include "llama_msgs/msg/chat_req_tool.h"
#include "llama_msgs/msg/chat_tool_call.h"
#include "llama_ros/llama.hpp"

common_chat_tool_choice llama_utils::parse_chat_tool_choice(int type) {
  if (type == llama_msgs::msg::ChatTool::TOOL_CHOICE_AUTO) {
    return COMMON_CHAT_TOOL_CHOICE_AUTO;
  } else if (type == llama_msgs::msg::ChatTool::TOOL_CHOICE_REQUIRED) {
    return COMMON_CHAT_TOOL_CHOICE_REQUIRED;
  } else if (type == llama_msgs::msg::ChatTool::TOOL_CHOICE_NONE) {
    return COMMON_CHAT_TOOL_CHOICE_NONE;
  } else {
    throw std::runtime_error("Unsupported chat tool choice: " +
                             std::to_string(type));
  }
}

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

  std::vector<common_chat_tool> tools;
  for (auto tool : goal->tools) {
    struct common_chat_tool t;
    t.name = tool.function.name;
    t.description = tool.function.description;
    t.parameters = tool.function.parameters;
    tools.push_back(t);
  }

  inputs.messages = messages;
  inputs.tools = tools;
  inputs.grammar = goal->sampling_config.grammar;
  inputs.json_schema = goal->sampling_config.grammar_schema;
  inputs.add_generation_prompt = goal->add_generation_prompt;
  inputs.use_jinja = goal->use_jinja;
  inputs.tool_choice = llama_utils::parse_chat_tool_choice(goal->tool_choice);
  inputs.parallel_tool_calls = goal->parallel_tool_calls;

  return inputs;
}

llama_msgs::action::GenerateChatCompletions::Result
llama_utils::generate_chat_completions_result(const ResponseResult &result) {
  std::string finish_reason = "length";
  common_chat_msg msg;

  if (result.stop == llama_ros::FULL_STOP || result.stop == llama_ros::ABORT) {
    msg = common_chat_parse(result.content, result.oaicompat_chat_format);
    finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
  } else {
    msg.content = result.content;
  }

  llama_msgs::msg::ChatMessage chat_msg;

  if (!msg.reasoning_content.empty()) {
    chat_msg.reasoning_content = msg.reasoning_content;
  }
  if (!msg.content.empty() || msg.tool_calls.empty()) {
    chat_msg.content = msg.content;
  }
  if (!msg.tool_calls.empty()) {
    std::vector<llama_msgs::msg::ChatToolCall> tool_calls;
    for (const auto &tc : msg.tool_calls) {
      llama_msgs::msg::ChatToolCall tool_call;
      tool_call.name = tc.name;
      tool_call.arguments = tc.arguments;
      tool_call.id = "call_" + llama_utils::random_string(8);
      tool_calls.push_back(tool_call);
    }
    chat_msg.tool_calls = tool_calls;
    chat_msg.role = msg.role;
  }

  llama_msgs::msg::ChatChoice choice;
  choice.finish_reason = finish_reason;
  choice.index = 0;
  choice.message = chat_msg;

  if (!result.stream && result.probs_output.size() > 0) {
    for (const auto &prob : result.probs_output) {
      llama_msgs::msg::TokenProbArray probs_msg;
      probs_msg.chosen_token = prob.chosen_token.token;
      for (const auto &p : prob.data) {
        llama_msgs::msg::TokenProb aux;
        aux.token = p.token;
        aux.probability = p.probability;
        aux.token_text = p.text;
        probs_msg.data.push_back(aux);
      }
      choice.logprobs.push_back(probs_msg);
    }
  }

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

std::vector<llama_msgs::action::GenerateChatCompletions::Feedback>
llama_utils::generate_chat_completions_feedback(const ResponseResult &result) {
  bool first = result.n_decoded == 0;

  std::vector<llama_msgs::msg::ChatChoiceChunk> choices;

  if (!first) {
    llama_msgs::msg::ChatChoiceChunk choice;
    choice.finish_reason = "";
    choice.index = 0;
    choice.delta.role = "assistant";
    choice.delta.content = result.content;
    choices.push_back(choice);
  } else {
    if (result.content.empty()) {
      auto choice = llama_msgs::msg::ChatChoiceChunk();
      choice.finish_reason = "";
      choice.index = 0;
      choice.delta.role = "assistant";
      choices.push_back(choice);
    } else {
      llama_msgs::action::GenerateChatCompletions::Feedback first_ret;
      first_ret.created = std::time(0);
      first_ret.object = "chat.completion.chunk";
      first_ret.model = result.oaicompat_model;
      first_ret.usage.completion_tokens = result.n_decoded;
      first_ret.usage.prompt_tokens = result.n_prompt_tokens;
      first_ret.usage.total_tokens = result.n_decoded + result.n_prompt_tokens;
      first_ret.system_fingerprint = result.build_info;

      llama_msgs::msg::ChatChoiceChunk choice;
      choice.finish_reason = "";
      choice.index = 0;
      choice.delta.role = "assistant";
      first_ret.choices.push_back(choice);

      llama_msgs::action::GenerateChatCompletions::Feedback second_ret;
      second_ret.created = std::time(0);
      second_ret.object = "chat.completion.chunk";
      second_ret.model = result.oaicompat_model;
      second_ret.usage.completion_tokens = result.n_decoded;
      second_ret.usage.prompt_tokens = result.n_prompt_tokens;
      second_ret.usage.total_tokens = result.n_decoded + result.n_prompt_tokens;
      second_ret.system_fingerprint = result.build_info;

      llama_msgs::msg::ChatChoiceChunk choice2;
      choice2.finish_reason = "";
      choice2.index = 0;
      choice2.delta.role = "assistant";
      choice2.delta.content = result.content;

      second_ret.choices.push_back(choice2);
      return {first_ret, second_ret};
    }
  }

  llama_msgs::action::GenerateChatCompletions::Feedback ret;
  ret.created = std::time(0);
  ret.object = "chat.completion.chunk";
  ret.model = result.oaicompat_model;
  ret.usage.completion_tokens = result.n_decoded;
  ret.usage.prompt_tokens = result.n_prompt_tokens;
  ret.usage.total_tokens = result.n_decoded + result.n_prompt_tokens;
  ret.system_fingerprint = result.build_info;

  if (result.probs_output.size() > 0) {
    auto prob = result.probs_output[0];
    llama_msgs::msg::TokenProbArray probs_msg;
    probs_msg.chosen_token = prob.chosen_token.token;
    for (const auto &p : prob.data) {
      llama_msgs::msg::TokenProb aux;
      aux.token = p.token;
      aux.probability = p.probability;
      aux.token_text = p.text;
      probs_msg.data.push_back(aux);
    }
    choices[0].logprobs = probs_msg;
  }

  ret.choices = choices;
  return {ret};
}
