

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include <memory>
#include <rclcpp/rclcpp.hpp>

#include "llama.h"
#include "llama_exception.hpp"
#include "llama_node.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

LlamaNode::LlamaNode() : rclcpp::Node("llama_node") {

  std::string prompt;

  // node params from llama.cpp common.h
  this->declare_parameters<int32_t>(
      "", {
              {"seed", -1},
              {"n_threads",
               std::min(4, (int32_t)std::thread::hardware_concurrency())},
              {"n_predict", 128},
              {"repeat_last_n", 64},
              {"n_parts", -1},
              {"n_ctx", 512},
              {"n_batch", 8},
              {"n_keep", 0},
              {"top_k", 40},
          });
  this->declare_parameters<std::string>(
      "",
      {
          {"model", std::string("/home/miguel/llama.cpp/models/alpaca/"
                                "ggml-alpaca-7b-q4-ggjt.bin")},
          {"prompt", "Below is an instruction that describes a task. Write a "
                     "response that appropriately completes the request."},
          {"input_prefix", ""},
      });
  this->declare_parameters<float>("", {
                                          {"top_p", 0.95f},
                                          {"temp", 0.80f},
                                          {"repeat_penalty", 1.10f},
                                      });
  this->declare_parameters<bool>("", {
                                         {"memory_f16", true},
                                         {"interactive", false},
                                         {"interactive_start", false},
                                         {"instruct", true},
                                         {"ignore_eos", false},
                                         {"use_mlock", false},
                                         {"verbose_prompt", false},
                                     });

  this->get_parameter("seed", this->seed);
  this->get_parameter("n_threads", this->n_threads);
  this->get_parameter("n_predict", this->n_predict);
  this->get_parameter("repeat_last_n", this->repeat_last_n);
  this->get_parameter("n_parts", this->n_parts);
  this->get_parameter("n_ctx", this->n_ctx);
  this->get_parameter("n_batch", this->n_batch);
  this->get_parameter("n_keep", this->n_keep);
  this->get_parameter("top_k", this->top_k);

  this->get_parameter("model", this->model);
  this->get_parameter("prompt", prompt);
  this->get_parameter("input_prefix", this->input_prefix);

  this->get_parameter("temp", this->temp);
  this->get_parameter("top_p", this->top_p);
  this->get_parameter("repeat_penalty", this->repeat_penalty);

  this->get_parameter("memory_f16", this->memory_f16);
  this->get_parameter("interactive", this->interactive);
  this->get_parameter("interactive_start", this->interactive_start);
  this->get_parameter("instruct", this->instruct);
  this->get_parameter("ignore_eos", this->ignore_eos);
  this->get_parameter("use_mlock", this->use_mlock);
  this->get_parameter("verbose_prompt", this->verbose_prompt);

  if (this->n_ctx > 2048) {
    RCLCPP_WARN(this->get_logger(),
                "Model does not support context sizes greater than 2048 tokens "
                "(%d specified); expect poor results",
                this->n_ctx);
  }

  // load the model
  auto lparams = llama_context_default_params();

  lparams.n_ctx = this->n_ctx;
  lparams.n_parts = this->n_parts;
  lparams.seed = this->seed;
  lparams.f16_kv = this->memory_f16;
  lparams.use_mlock = this->use_mlock;

  this->ctx = llama_init_from_file(this->model.c_str(), lparams);
  this->n_ctx = llama_n_ctx(this->ctx);

  if (this->ctx == NULL) {
    RCLCPP_ERROR(this->get_logger(), "failed to load model '%s'",
                 this->model.c_str());
    throw LlamaException("failed to load model " + this->model);
  }

  // show system information
  RCLCPP_INFO(this->get_logger(), "system_info: n_threads = %d / %d | %s",
              this->n_threads, std::thread::hardware_concurrency(),
              llama_print_system_info());

  this->process_initial_prompt(prompt);

  // prefix & suffix for instruct mode
  this->inp_pfx =
      this->llama_node_tokenize(this->ctx, "\n\n### Instruction:\n\n", true);
  this->inp_sfx =
      this->llama_node_tokenize(this->ctx, "\n\n### Response:\n\n", false);

  // in instruct mode, we inject a prefix and a suffix to each input by the user
  if (this->instruct) {
    this->interactive_start = true;
    this->antiprompt.push_back("### Instruction:\n\n");
  }

  // enable interactive mode if reverse prompt or interactive start is specified
  if (this->antiprompt.size() != 0 || this->interactive_start) {
    this->interactive = true;
  }

  // determine newline token
  this->llama_token_newline = this->llama_node_tokenize(ctx, "\n", false);

  if (this->interactive) {
    RCLCPP_INFO(this->get_logger(), "interactive mode on");

    if (this->antiprompt.size()) {
      for (auto antiprompt : this->antiprompt) {
        RCLCPP_INFO(this->get_logger(), "reverse prompt: '%s'",
                    antiprompt.c_str());
      }
    }

    if (!this->input_prefix.empty()) {
      RCLCPP_INFO(this->get_logger(), "input prefix: '%s'",
                  this->input_prefix.c_str());
    }
  }

  // show info
  RCLCPP_INFO(this->get_logger(),
              "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = "
              "%i, repeat_penalty = %f",
              this->temp, this->top_k, this->top_p, this->repeat_last_n,
              this->repeat_penalty);
  RCLCPP_INFO(
      this->get_logger(),
      "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n",
      n_ctx, this->n_batch, this->n_predict, this->n_keep);

  if (this->interactive) {
    this->is_interacting = this->interactive_start;
  }

  this->gpt_service = this->create_service<llama_msgs::srv::GPT>(
      "gpt", std::bind(&LlamaNode::gpt_cb, this, _1, _2));
}

LlamaNode::~LlamaNode() {
  llama_print_timings(this->ctx);
  llama_free(this->ctx);
}

void LlamaNode::gpt_cb(
    const std::shared_ptr<llama_msgs::srv::GPT::Request> request,
    std::shared_ptr<llama_msgs::srv::GPT::Response> response) {

  std::string buffer = request->prompt;

  if (buffer.length() > 1) {

    // instruct mode: insert instruction prefix
    if (this->instruct && !this->is_antiprompt) {
      this->n_consumed = this->embd_inp.size();
      this->embd_inp.insert(this->embd_inp.end(), this->inp_pfx.begin(),
                            this->inp_pfx.end());
    }

    auto line_inp = this->llama_node_tokenize(this->ctx, buffer, false);
    this->embd_inp.insert(this->embd_inp.end(), line_inp.begin(),
                          line_inp.end());

    // instruct mode: insert response suffix
    if (this->instruct) {
      this->embd_inp.insert(this->embd_inp.end(), this->inp_sfx.begin(),
                            this->inp_sfx.end());
    }

    this->n_remain -= line_inp.size();
  }

  this->input_noecho = true; // do not echo this again

  response->response = this->process_prompt();
}

std::vector<llama_token>
LlamaNode::llama_node_tokenize(struct llama_context *ctx,
                               const std::string &text, bool add_bos) {
  // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
  std::vector<llama_token> res(text.size() + (int)add_bos);
  int n = llama_tokenize(ctx, text.c_str(), res.data(), res.size(), add_bos);
  assert(n >= 0);
  res.resize(n);

  return res;
}

void LlamaNode::process_initial_prompt(std::string prompt) {

  // Add a space in front of the first character to match OG llama tokenizer
  // behavior
  prompt.insert(0, 1, ' ');

  // tokenize the prompt
  this->embd_inp = this->llama_node_tokenize(this->ctx, prompt, true);

  if ((int)embd_inp.size() > this->n_ctx - 4) {
    RCLCPP_ERROR(this->get_logger(), "prompt is too long (%d tokens, max %d)",
                 (int)this->embd_inp.size(), this->n_ctx - 4);
    throw LlamaException("prompt is too long (" +
                         std::to_string((int)this->embd_inp.size()) +
                         " tokens, max " + std::to_string(this->n_ctx) + ")");
  }

  // number of tokens to keep when resetting context
  if (this->n_keep < 0 || this->n_keep > (int)this->embd_inp.size() ||
      this->instruct) {
    this->n_keep = (int)this->embd_inp.size();
  }

  this->process_prompt();
}

std::string LlamaNode::process_prompt() {

  // TODO: replace with ring-buffer
  std::vector<llama_token> last_n_tokens(this->n_ctx);
  std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

  std::vector<llama_token> embd;
  std::string result;

  while (this->n_remain != 0 || this->interactive) {

    // predict
    if (embd.size() > 0) {
      // infinite text generation via context swapping
      // if we run out of context:
      // - take the n_keep first tokens from the original prompt (via n_past)
      // - take half of the last (n_ctx - n_keep) tokens and recompute the
      // logits in a batch
      if (n_past + (int)embd.size() > n_ctx) {

        const int n_left = n_past - this->n_keep;
        n_past = this->n_keep;

        // insert n_left/2 tokens at the start of embd from last_n_tokens
        embd.insert(embd.begin(),
                    last_n_tokens.begin() + n_ctx - n_left / 2 - embd.size(),
                    last_n_tokens.end() - embd.size());
      }

      if (llama_eval(ctx, embd.data(), embd.size(), n_past, this->n_threads)) {
        RCLCPP_ERROR(this->get_logger(), "failed to eval");
        throw LlamaException("failed to eval");
      }
    }

    n_past += embd.size();
    embd.clear();

    if ((int)embd_inp.size() <= this->n_consumed && !this->is_interacting) {
      // out of user input, sample next token

      llama_token id = 0;

      {
        auto logits = llama_get_logits(this->ctx);

        if (this->ignore_eos) {
          logits[llama_token_eos()] = 0;
        }

        id = llama_sample_top_p_top_k(
            this->ctx, last_n_tokens.data() + n_ctx - this->repeat_last_n,
            this->repeat_last_n, this->top_k, this->top_p, this->temp,
            this->repeat_penalty);

        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token when in interactive mode
      if (id == llama_token_eos() && this->interactive && !this->instruct) {
        id = this->llama_token_newline.front();
        if (this->antiprompt.size() != 0) {
          // tokenize and inject first reverse prompt
          const auto first_antiprompt =
              this->llama_node_tokenize(ctx, this->antiprompt.front(), false);
          this->embd_inp.insert(this->embd_inp.end(), first_antiprompt.begin(),
                                first_antiprompt.end());
        }
      }

      // add it to the context
      embd.push_back(id);

      // echo this to console
      input_noecho = false;

      // decrement remaining sampling budget
      --n_remain;

    } else {
      // some user input remains from prompt or interaction, forward it to
      // processing
      while ((int)embd_inp.size() > n_consumed) {
        embd.push_back(embd_inp[n_consumed]);
        last_n_tokens.erase(last_n_tokens.begin());
        last_n_tokens.push_back(embd_inp[n_consumed]);
        ++n_consumed;
        if ((int)embd.size() >= this->n_batch) {
          break;
        }
      }
    }

    // display text
    if (!input_noecho) {
      for (auto id : embd) {
        std::string aux_s = llama_token_to_str(this->ctx, id);
        result.append(aux_s);
        RCLCPP_INFO(this->get_logger(), "%s", aux_s.c_str());
      }
    }

    // in interactive mode, and not currently processing queued inputs;
    // check if we should prompt the user for more
    if (this->interactive && (int)this->embd_inp.size() <= this->n_consumed) {

      // check for reverse prompt
      if (this->antiprompt.size()) {
        std::string last_output;
        for (auto id : last_n_tokens) {
          last_output += llama_token_to_str(ctx, id);
        }

        is_antiprompt = false;
        // Check if each of the reverse prompts appears at the end of the
        // output.
        for (std::string &antiprompt : this->antiprompt) {
          if (last_output.find(antiprompt.c_str(),
                               last_output.length() - antiprompt.length(),
                               antiprompt.length()) != std::string::npos) {
            this->is_interacting = true;
            is_antiprompt = true;
            // set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);
            fflush(stdout);
            break;
          }
        }
      }

      if (this->n_past > 0 && this->is_interacting) {
        return result;
      }

      if (n_past > 0) {
        this->is_interacting = false;
      }
    }

    // end of text token
    if (embd.back() == llama_token_eos()) {
      if (this->instruct) {
        this->is_interacting = true;
      }
    }

    // In interactive mode, respect the maximum number of tokens and drop back
    // to user input when reached.
    if (this->interactive && this->n_remain <= 0 && this->n_predict != -1) {
      this->n_remain = this->n_predict;
      this->is_interacting = true;
    }
  }

  return "";
}

int main(int argc, char *argv[]) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LlamaNode>());
  rclcpp::shutdown();
  return 0;
}