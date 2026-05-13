// MIT License
//
// Copyright (c) 2025 Miguel Ángel González Santamarta
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

#include <common.h>

#include <cstddef>
#include <mutex>
#include <regex>

#include "llama_ros/llama.hpp"
#include "llama_utils/chat_utils.hpp"
#include "llama_utils/llama_params.hpp"

#include <llama_msgs/action/generate_response.hpp>
#include <llama_msgs/msg/detail/chat_req_tool__struct.hpp>

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

common_reasoning_format
llama_utils::parse_reasoning_format(const int reasoning_format) {
  if (reasoning_format ==
      llama_msgs::msg::ChatReasoningFormat::COMMON_REASONING_FORMAT_DEEPSEEK) {
    return COMMON_REASONING_FORMAT_DEEPSEEK;
  } else if (reasoning_format == llama_msgs::msg::ChatReasoningFormat::
                                     COMMON_REASONING_FORMAT_AUTO) {
    return COMMON_REASONING_FORMAT_AUTO;
  } else if (reasoning_format == llama_msgs::msg::ChatReasoningFormat::
                                     COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY) {
    return COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
  } else {
    return COMMON_REASONING_FORMAT_NONE;
  }
}

common_chat_templates_inputs llama_utils::parse_chat_completions_goal(
    const std::shared_ptr<
        const llama_msgs::action::GenerateChatCompletions::Goal>
        goal) {

  common_chat_templates_inputs inputs;

  std::vector<common_chat_msg> messages;
  for (auto message : goal->messages) {
    common_chat_msg msg;
    msg.role = message.role;
    msg.content = message.content;
    msg.tool_call_id = message.tool_call_id;    
    std::vector<common_chat_msg_content_part> content_parts;

    for (auto content_part : message.content_parts) {
      common_chat_msg_content_part part;
      part.type = content_part.type;
      part.text = content_part.text;
      content_parts.push_back(part);
    }
    msg.content_parts = content_parts;

    std::vector<common_chat_tool_call> tool_calls;
    for (auto tool_call : message.tool_calls) {
      common_chat_tool_call call;
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
    common_chat_tool t;
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
  inputs.reasoning_format =
      llama_utils::parse_reasoning_format(goal->reasoning_format.value);
  inputs.enable_thinking =
      inputs.reasoning_format != COMMON_REASONING_FORMAT_NONE;
  inputs.force_pure_content = goal->force_pure_content_parser;

  return inputs;
}

llama_msgs::action::GenerateChatCompletions::Result
llama_utils::generate_chat_completions_result(
    const llama_ros::ServerTaskResultCompletion &result) {
  llama_msgs::msg::ChatMessage chat_msg;
  std::string finish_reason = "stop";

  common_chat_msg msg = result.oaicompat_msg;

  if (msg.tool_calls.size() > 0) {
    finish_reason = "tool_calls";
  }

  chat_msg.role = msg.role;
  if (!msg.reasoning_content.empty()) {
    chat_msg.reasoning_content = msg.reasoning_content;
  }

  if (!msg.content.empty() || msg.tool_calls.empty()) {
    chat_msg.content = msg.content;
  }

  if (!msg.tool_calls.empty()) {
    std::vector<llama_msgs::msg::ChatToolCall> tool_calls;
    for (size_t i = 0; i < msg.tool_calls.size(); ++i) {
      const auto &tc = msg.tool_calls[i];
      llama_msgs::msg::ChatToolCall tool_call;
      tool_call.name = tc.name;
      tool_call.arguments = tc.arguments;
      tool_call.id = "call_" + llama_utils::random_string(8);
      tool_call.index = i;
      tool_calls.push_back(tool_call);
    }
    chat_msg.tool_calls = tool_calls;
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
llama_utils::generate_chat_completions_feedback(
    const llama_ros::ServerTaskResultCompletionPartial &result,
    std::vector<common_chat_msg_diff> deltas,
    std::vector<llama_msgs::msg::TokenProb> probs) {
  bool first = result.n_decoded == 0;

  std::vector<llama_msgs::action::GenerateChatCompletions::Feedback> feedbacks;
  std::vector<llama_msgs::msg::ChatChoiceChunk> choices;

  if (first) {
    llama_msgs::action::GenerateChatCompletions::Feedback first_msg;
    first_msg.created = std::time(0);
    first_msg.object = "chat.completion.chunk";
    first_msg.model = result.oaicompat_model;
    first_msg.id = result.oaicompat_cmpl_id;
    first_msg.system_fingerprint = result.build_info;

    llama_msgs::msg::ChatChoiceChunk choice;
    choice.delta.role = "assistant";
    choice.index = 0;
    choice.finish_reason = "";
    choice.delta.content = "";

    first_msg.choices.push_back(choice);
    feedbacks.push_back(first_msg);
  }

  for (auto &diff : deltas) {
    llama_msgs::action::GenerateChatCompletions::Feedback feedback;
    feedback.created = std::time(0);
    feedback.object = "chat.completion.chunk";
    feedback.model = result.oaicompat_model;
    feedback.id = result.oaicompat_cmpl_id;
    feedback.system_fingerprint = result.build_info;
    feedback.usage.completion_tokens = result.n_decoded;
    feedback.usage.prompt_tokens = result.n_prompt_tokens;
    feedback.usage.total_tokens = result.n_decoded + result.n_prompt_tokens;

    llama_msgs::msg::ChatChoiceChunk choice;
    choice.index = 0;
    choice.finish_reason = "";
    choice.delta.role = "assistant";

    if (!diff.reasoning_content_delta.empty()) {
      choice.delta.reasoning_content = diff.reasoning_content_delta;
    }
    if (!diff.content_delta.empty()) {
      choice.delta.content = diff.content_delta;
    }
    if (diff.tool_call_index != std::string::npos) {
      llama_msgs::msg::ChatToolCall tool_call;

      tool_call.index = diff.tool_call_index;
      tool_call.id = diff.tool_call_delta.id;
      tool_call.arguments = diff.tool_call_delta.arguments;
      tool_call.name = diff.tool_call_delta.name;

      choice.delta.tool_calls.push_back(tool_call);
    }

    // Add logprobs to the streaming feedback if available
    if (!probs.empty()) {
      llama_msgs::msg::TokenProbArray logprobs_msg;
      logprobs_msg.chosen_token = probs[0].token;
      for (const auto &prob : probs) {
        llama_msgs::msg::TokenProb aux;
        aux.token = prob.token;
        // Convert probability to log probability
        aux.probability = std::log(prob.probability);
        aux.token_text = prob.token_text;
        logprobs_msg.data.push_back(aux);
      }
      choice.logprobs = logprobs_msg;
    }

    feedback.choices.push_back(choice);
    feedbacks.push_back(feedback);
  }

  return feedbacks;
}

llama_utils::ChatCompletionsContext llama_utils::prepare_chat_completions_call(
    const std::shared_ptr<
        const llama_msgs::action::GenerateChatCompletions::Goal> &goal,
    llama_ros::Llama *llama) {
  llama_utils::ChatCompletionsContext ctx;

  // Get model chat template
  auto tmpls = llama->get_chat_formatter();
  ctx.prompt_format_config = llama_utils::parse_chat_completions_goal(goal);
  ctx.chat_prompt_instance =
      llama->get_chat_params(tmpls->get_templates(), ctx.prompt_format_config);
  ctx.sparams = llama_utils::parse_sampling_params(goal->sampling_config,
                                                   llama->get_n_vocab());

  ctx.oaicompat_chat_syntax.format = ctx.chat_prompt_instance.format;
  ctx.oaicompat_chat_syntax.reasoning_format =
      llama_utils::parse_reasoning_format(goal->reasoning_format.value);
  ctx.oaicompat_chat_syntax.reasoning_in_content =
      goal->stream && ctx.oaicompat_chat_syntax.reasoning_format ==
                          COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY;
  ctx.oaicompat_chat_syntax.generation_prompt =
      ctx.chat_prompt_instance.generation_prompt;
  ctx.oaicompat_chat_syntax.parse_tool_calls =
      !goal->tools.empty() &&
      ctx.prompt_format_config.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

  // Load the PEG parser from the chat template so that common_chat_parse
  // can properly extract tool calls and structured output from the response.
  if (!ctx.chat_prompt_instance.parser.empty()) {
    ctx.oaicompat_chat_syntax.parser.load(ctx.chat_prompt_instance.parser);
  }

  // Add preserved tokens to the sampling config.
  // These are special tokens required by the chat template parser
  // (e.g. <tool_call>, </tool_call>) that must be tokenized as single tokens.
  if (llama) {
    const llama_vocab *vocab = llama->get_vocab();
    for (const auto &token_str : ctx.chat_prompt_instance.preserved_tokens) {
      auto ids = common_tokenize(vocab, token_str,
                                 /* add_special= */ false,
                                 /* parse_special= */ true);
      if (ids.size() == 1) {
        ctx.sparams.preserved_tokens.insert(ids[0]);
      }
    }
  }

  if (goal->sampling_config.grammar.empty() && goal->tools.size() != 0) {
    if (!ctx.chat_prompt_instance.grammar.empty()) {
      ctx.sparams.grammar = {COMMON_GRAMMAR_TYPE_TOOL_CALLS,
                             ctx.chat_prompt_instance.grammar};
    }

    // Workaround for llama.cpp autoparser bug (b8261): when a template uses
    // per-call wrapping tags (e.g. <tool_call>...</tool_call> for each call,
    // as in Qwen3) but the JSON inside uses standard format, the autoparser
    // detects it as JSON_NATIVE and puts ALL calls inside a SINGLE tag pair
    // with comma separation. This is wrong — the model expects each call in
    // its own tag pair. Fix the grammar when parallel_tool_calls is true by
    // rewriting the tool-call rule so each call gets its own wrapper.
    if (ctx.prompt_format_config.parallel_tool_calls &&
        !ctx.sparams.grammar.empty()) {
      // Look for the tool-call rule that has comma-separated repetition
      // inside a single pair of XML-like tags. Pattern in the grammar:
      //   tool-call ::= "OPEN" space CHOICES (space "," space CHOICES)* space
      //   "CLOSE"
      // We rewrite to:
      //   tool-call ::= "OPEN" space CHOICES space "CLOSE" (space "OPEN" space
      //   CHOICES space "CLOSE")*
      std::string &grammar = ctx.sparams.grammar.grammar;

      // Find the tool-call rule
      std::string rule_marker = "tool-call ::= ";
      auto rule_pos = grammar.find(rule_marker);
      if (rule_pos != std::string::npos) {
        // Find the end of this rule (next rule or end of string)
        auto rule_start = rule_pos + rule_marker.size();
        auto next_rule = grammar.find('\n', rule_start);
        if (next_rule == std::string::npos)
          next_rule = grammar.size();

        std::string rule_body =
            grammar.substr(rule_start, next_rule - rule_start);

        // Check if this rule has the comma-separated pattern inside XML tags.
        // Look for: (space "," space CHOICES)*
        auto comma_pat_pos = rule_body.find("(space \",\" space ");
        // Also check for opening XML tag pattern: starts with "< ...
        bool has_xml_open =
            rule_body.size() > 1 && rule_body[0] == '"' && rule_body[1] == '<';

        if (comma_pat_pos != std::string::npos && has_xml_open) {
          // Extract the opening tag: everything from start to first " space "
          // e.g. "<tool_call>\n" space
          // Find the first occurrence of '" space' after the opening quote
          auto space_after_open = rule_body.find("\" space ");
          if (space_after_open != std::string::npos) {
            // Opening tag includes up to and including ' space '
            std::string open_tag =
                rule_body.substr(0, space_after_open + 8); // +8 for `" space `

            // Extract the choices group between open_tag and comma pattern
            std::string choices = rule_body.substr(
                open_tag.size(), comma_pat_pos - open_tag.size());
            // Trim trailing whitespace from choices
            while (!choices.empty() &&
                   (choices.back() == ' ' || choices.back() == '\t'))
              choices.pop_back();

            // Extract closing tag: after the comma-repetition pattern
            // The pattern ends with ")* space CLOSE"
            // Find ")* " after comma_pat_pos
            auto rep_end = rule_body.find(")*", comma_pat_pos);
            if (rep_end != std::string::npos) {
              std::string close_tag =
                  rule_body.substr(rep_end + 2); // after ")*"
              // Trim leading whitespace
              while (!close_tag.empty() &&
                     (close_tag.front() == ' ' || close_tag.front() == '\t'))
                close_tag.erase(close_tag.begin());

              // Build the corrected rule:
              // tool-call ::= OPEN CHOICES CLOSE (space OPEN CHOICES CLOSE)*
              std::string one_call = open_tag + choices + " " + close_tag;
              std::string fixed_rule = one_call + " (space " + one_call + ")*";

              grammar.replace(rule_start, next_rule - rule_start, fixed_rule);
            }
          }
        }
      }
    }
  }

  if (goal->sampling_config.grammar_triggers.empty()) {
    // Resolve WORD triggers to TOKEN triggers when possible,
    // matching the llama.cpp server behavior.
    for (auto trigger : ctx.chat_prompt_instance.grammar_triggers) {
      if (trigger.type == COMMON_GRAMMAR_TRIGGER_TYPE_WORD && llama) {
        const llama_vocab *vocab = llama->get_vocab();
        auto ids = common_tokenize(vocab, trigger.value,
                                   /* add_special= */ false,
                                   /* parse_special= */ true);
        if (ids.size() == 1) {
          trigger.type = COMMON_GRAMMAR_TRIGGER_TYPE_TOKEN;
          trigger.token = ids[0];
        }
      }
      ctx.sparams.grammar_triggers.push_back(trigger);
    }
  }

  ctx.sparams.grammar_lazy = ctx.chat_prompt_instance.grammar_lazy ||
                             goal->sampling_config.grammar_lazy;

  // Set up reasoning budget vocab tokens if budget is active
  if (ctx.sparams.reasoning_budget_tokens >= 0 &&
      !ctx.chat_prompt_instance.thinking_end_tag.empty() && llama) {
    const llama_vocab *vocab = llama->get_vocab();
    if (ctx.sparams.reasoning_budget_start.empty() &&
        !ctx.chat_prompt_instance.thinking_start_tag.empty()) {
      ctx.sparams.reasoning_budget_start =
          common_tokenize(vocab, ctx.chat_prompt_instance.thinking_start_tag,
                          /* add_special= */ false, /* parse_special= */ true);
    }
    if (ctx.sparams.reasoning_budget_end.empty()) {
      ctx.sparams.reasoning_budget_end =
          common_tokenize(vocab, ctx.chat_prompt_instance.thinking_end_tag,
                          /* add_special= */ false, /* parse_special= */ true);
    }

    if (ctx.sparams.reasoning_budget_forced.empty()) {
      ctx.sparams.reasoning_budget_forced =
          common_tokenize(vocab,
                          ctx.sparams.reasoning_budget_message +
                              ctx.chat_prompt_instance.thinking_end_tag,
                          /* add_special= */ false, /* parse_special= */ true);
    }
  }

  return ctx;
}

llama_utils::CompletionContext llama_utils::prepare_completion_call(
    const std::shared_ptr<const llama_msgs::action::GenerateResponse::Goal>
        &goal,
    llama_ros::Llama *llama) {
  llama_utils::CompletionContext ctx;

  ctx.prompt = goal->prompt;
  ctx.stop = goal->stop;
  ctx.reset = goal->reset;

  // Apply EOG logit biases to sampling configuration
  llama_msgs::msg::SamplingConfig sampling_config = goal->sampling_config;

  if (llama && llama->get_vocab() && llama->get_ctx()) {
    llama_utils::apply_eog_logit_biases(sampling_config, llama->get_vocab(),
                                        llama->get_ctx());
  }

  ctx.sparams = llama_utils::parse_sampling_params(
      sampling_config, llama ? llama->get_n_vocab() : 0);

  return ctx;
}

llama_msgs::action::GenerateResponse::Result
llama_utils::generate_completion_result(
    const llama_ros::ServerTaskResultCompletion &result,
    llama_ros::Llama *llama) {
  llama_msgs::action::GenerateResponse::Result ros_result;

  ros_result.response.text = result.content;
  ros_result.response.tokens = result.tokens;

  if (llama) {
    for (const auto &probs_msg : result.probs_output) {
      llama_msgs::msg::TokenProbArray probs_msg_aux;
      for (const auto &prob : probs_msg.data) {
        llama_msgs::msg::TokenProb aux;
        aux.token = prob.token;
        aux.probability = prob.probability;
        aux.token_text =
            llama->detokenize(std::vector<llama_token>{prob.token});
        probs_msg_aux.data.push_back(aux);
      }
      ros_result.response.probs.push_back(probs_msg_aux);
    }
  }

  return ros_result;
}

llama_msgs::action::GenerateResponse::Feedback
llama_utils::create_completion_feedback(
    const llama_ros::CompletionOutput &completion, llama_ros::Llama *llama) {
  llama_msgs::action::GenerateResponse::Feedback feedback;

  if (llama) {
    feedback.partial_response.text = llama->detokenize({completion.token});
  }
  feedback.partial_response.token = completion.token;
  feedback.partial_response.probs.chosen_token = completion.token;

  if (llama) {
    for (auto prob : completion.probs) {
      llama_msgs::msg::TokenProb aux;
      aux.token = prob.token;
      aux.probability = prob.probability;
      aux.token_text = llama->detokenize({prob.token});
      feedback.partial_response.probs.data.push_back(aux);
    }
  }

  return feedback;
}

int32_t llama_utils::uuid_to_int32(const std::array<uint8_t, 16> &uuid) {
  int32_t value;
  std::memcpy(&value, uuid.data(), sizeof(int32_t));
  return value;
}

uint64_t llama_utils::generate_random_uint64() {
  static std::random_device rd;
  static std::mt19937_64 eng(rd());
  static std::uniform_int_distribution<uint64_t> distr;
  static std::mutex rng_mutex;

  std::lock_guard<std::mutex> lock(rng_mutex);
  return distr(eng);
}
