
#include <fstream>
#include <memory>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "llama.h"
#include "llama_ros/llama_exception.hpp"
#include "llama_ros/llama_node.hpp"

using namespace llama_ros;
using std::placeholders::_1;
using std::placeholders::_2;

LlamaNode::LlamaNode() : rclcpp::Node("llama_node") {

  std::string prompt = "";
  std::string file_path;

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
      "", {
              {"model", std::string("models/lamma-7B/ggml-model.bin")},
              {"prompt", ""},
              {"file", ""},
              {"input_prefix", ""},
          });
  this->declare_parameters<float>("", {
                                          {"top_p", 0.95f},
                                          {"temp", 0.80f},
                                          {"repeat_penalty", 1.10f},
                                      });
  this->declare_parameters<bool>("", {
                                         {"memory_f16", true},
                                         {"instruct", false},
                                         {"ignore_eos", false},
                                         {"use_mlock", false},
                                     });
  this->declare_parameter<std::vector<std::string>>("reverse_prompt",
                                                    std::vector<std::string>());

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
  this->get_parameter("file", file_path);
  this->get_parameter("input_prefix", this->input_prefix);

  this->get_parameter("temp", this->temp);
  this->get_parameter("top_p", this->top_p);
  this->get_parameter("repeat_penalty", this->repeat_penalty);

  this->get_parameter("memory_f16", this->memory_f16);
  this->get_parameter("instruct", this->instruct);
  this->get_parameter("ignore_eos", this->ignore_eos);
  this->get_parameter("use_mlock", this->use_mlock);

  this->get_parameter("reverse_prompt", this->antiprompt);

  if (this->antiprompt.front().empty() && this->antiprompt.size() == 1) {
    this->antiprompt.clear();
  }

  if (this->n_ctx > 2048) {
    RCLCPP_WARN(this->get_logger(),
                "Model does not support context sizes greater than 2048 tokens "
                "(%d specified); expect poor results",
                this->n_ctx);
  }

  // load prompt from file
  if (!file_path.empty()) {
    std::ifstream file(file_path.c_str());
    if (!file) {
      LlamaException("Failed to open file " + file_path);
    }

    std::copy(std::istreambuf_iterator<char>(file),
              std::istreambuf_iterator<char>(), back_inserter(prompt));
    if (prompt.back() == '\n') {
      prompt.pop_back();
    }
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
    RCLCPP_ERROR(this->get_logger(), "Failed to load model '%s'",
                 this->model.c_str());
    throw LlamaException("Failed to load model " + this->model);
  }

  // show system information
  RCLCPP_INFO(this->get_logger(), "System_info: n_threads = %d / %d | %s",
              this->n_threads, std::thread::hardware_concurrency(),
              llama_print_system_info());

  // prefix & suffix for instruct mode
  this->inp_pfx = this->tokenize("\n\n### Instruction:\n\n", true);
  this->inp_sfx = this->tokenize("\n\n### Response:\n\n", false);

  // in instruct mode, we inject a prefix and a suffix to each input by the user
  if (this->instruct) {
    this->antiprompt.push_back("### Instruction:");
  }

  // determine newline token
  this->llama_token_newline = this->tokenize("\n", false);

  // TODO: replace with ring-buffer
  this->last_n_tokens = std::vector<llama_token>(this->n_ctx);
  std::fill(this->last_n_tokens.begin(), this->last_n_tokens.end(), 0);

  this->n_past = 0;
  this->is_antiprompt = false;
  this->n_remain = this->n_predict;
  this->n_consumed = 0;

  // show info
  RCLCPP_INFO(this->get_logger(),
              "Sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = "
              "%i, repeat_penalty = %f",
              this->temp, this->top_k, this->top_p, this->repeat_last_n,
              this->repeat_penalty);
  RCLCPP_INFO(
      this->get_logger(),
      "Generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n",
      n_ctx, this->n_batch, this->n_predict, this->n_keep);

  this->process_initial_prompt(prompt);

  // srv and pub
  this->text_pub =
      this->create_publisher<std_msgs::msg::String>("gpt_text", 10);
  this->gpt_service = this->create_service<llama_msgs::srv::GPT>(
      "gpt", std::bind(&LlamaNode::gpt_cb, this, _1, _2));

  RCLCPP_INFO(this->get_logger(), "Llama Node started");
}

LlamaNode::~LlamaNode() { llama_free(this->ctx); }

void LlamaNode::gpt_cb(
    const std::shared_ptr<llama_msgs::srv::GPT::Request> request,
    std::shared_ptr<llama_msgs::srv::GPT::Response> response) {

  std::string buffer = request->prompt;
  RCLCPP_INFO(this->get_logger(), "Prompt received: %s", buffer.c_str());

  if (buffer.length() > 1) {

    if (this->input_prefix.empty()) {
      buffer = this->input_prefix + buffer;
    }

    // instruct mode: insert instruction prefix
    if (this->instruct && !this->is_antiprompt) {
      this->n_consumed = this->embd_inp.size();
      this->embd_inp.insert(this->embd_inp.end(), this->inp_pfx.begin(),
                            this->inp_pfx.end());
    }

    auto line_inp = this->tokenize(buffer, false);
    this->embd_inp.insert(this->embd_inp.end(), line_inp.begin(),
                          line_inp.end());

    // instruct mode: insert response suffix
    if (this->instruct) {
      this->embd_inp.insert(this->embd_inp.end(), this->inp_sfx.begin(),
                            this->inp_sfx.end());
    }

    this->n_remain -= line_inp.size();

    response->response = this->process_prompt(true);
  }
}

std::vector<llama_token> LlamaNode::tokenize(const std::string &text,
                                             bool add_bos) {
  // initialize to prompt numer of chars, since n_tokens <= n_prompt_chars
  std::vector<llama_token> res(text.size() + (int)add_bos);
  int n =
      llama_tokenize(this->ctx, text.c_str(), res.data(), res.size(), add_bos);
  assert(n >= 0);
  res.resize(n);

  return res;
}

void LlamaNode::process_initial_prompt(std::string prompt) {

  // Add a space in front of the first character to match OG llama tokenizer
  // behavior
  prompt.insert(0, 1, ' ');

  // tokenize the prompt
  this->embd_inp = this->tokenize(prompt, true);

  // number of tokens to keep when resetting context
  if (this->n_keep < 0 || this->n_keep > (int)this->embd_inp.size() ||
      this->instruct) {
    this->n_keep = (int)this->embd_inp.size();
  }

  if (prompt.length() > 1) {
    this->process_prompt(false);
  }
}

std::string LlamaNode::process_prompt(bool publish) {

  if ((int)this->embd_inp.size() > this->n_ctx - 4) {
    RCLCPP_ERROR(this->get_logger(), "Prompt is too long (%d tokens, max %d)",
                 (int)this->embd_inp.size(), this->n_ctx - 4);
    throw LlamaException("Prompt is too long (" +
                         std::to_string((int)this->embd_inp.size()) +
                         " tokens, max " + std::to_string(this->n_ctx) + ")");
  }

  bool input_noecho = true;
  std::string result;

  while (this->n_remain != 0) {

    // predict
    if (this->embd.size() > 0) {
      // infinite text generation via context swapping
      // if we run out of context:
      // - take the n_keep first tokens from the original prompt (via n_past)
      // - take half of the last (n_ctx - n_keep) tokens and recompute the
      // logits in a batch
      if (this->n_past + (int)this->embd.size() > this->n_ctx) {

        const int n_left = this->n_past - this->n_keep;
        this->n_past = this->n_keep;

        // insert n_left/2 tokens at the start of embd from last_n_tokens
        this->embd.insert(this->embd.begin(),
                          this->last_n_tokens.begin() + this->n_ctx -
                              n_left / 2 - this->embd.size(),
                          this->last_n_tokens.end() - this->embd.size());
      }

      if (llama_eval(this->ctx, this->embd.data(), this->embd.size(),
                     this->n_past, this->n_threads)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to eval");
        throw LlamaException("Failed to eval");
      }
    }

    this->n_past += this->embd.size();
    this->embd.clear();

    if ((int)this->embd_inp.size() <= this->n_consumed) {
      // out of user input, sample next token

      llama_token id = 0;

      {
        auto logits = llama_get_logits(this->ctx);

        if (this->ignore_eos) {
          logits[llama_token_eos()] = 0;
        }

        id = llama_sample_top_p_top_k(
            this->ctx, this->last_n_tokens.data() + n_ctx - this->repeat_last_n,
            this->repeat_last_n, this->top_k, this->top_p, this->temp,
            this->repeat_penalty);

        this->last_n_tokens.erase(this->last_n_tokens.begin());
        this->last_n_tokens.push_back(id);
      }

      // replace end of text token with newline token
      if (id == llama_token_eos() && !this->instruct) {
        id = this->llama_token_newline.front();
        if (this->antiprompt.size() != 0) {
          // tokenize and inject first reverse prompt
          const auto first_antiprompt =
              this->tokenize(this->antiprompt.front(), false);
          this->embd_inp.insert(this->embd_inp.end(), first_antiprompt.begin(),
                                first_antiprompt.end());
        }
      }

      // add it to the context
      this->embd.push_back(id);

      // echo this to console
      input_noecho = false;

      // decrement remaining sampling budget
      --this->n_remain;

    } else {
      // some user input remains from prompt, forward it to processing
      while ((int)this->embd_inp.size() > this->n_consumed) {
        this->embd.push_back(this->embd_inp[this->n_consumed]);
        this->last_n_tokens.erase(this->last_n_tokens.begin());
        this->last_n_tokens.push_back(this->embd_inp[this->n_consumed]);
        ++this->n_consumed;
        if ((int)this->embd.size() >= this->n_batch) {
          break;
        }
      }
    }

    // when not currently processing queued
    // inputs check if we should end
    if ((int)this->embd_inp.size() <= this->n_consumed) {

      // check for reverse prompt
      if (this->antiprompt.size()) {
        std::string last_output;
        for (auto id : this->last_n_tokens) {
          last_output += llama_token_to_str(this->ctx, id);
        }

        this->is_antiprompt = false;

        // check if each of the reverse prompts
        // appears at the end of the output.
        for (std::string &a : this->antiprompt) {
          if (last_output.find(a.c_str(), last_output.length() - a.length(),
                               a.length()) != std::string::npos) {
            this->is_antiprompt = true;
            break;
          }
        }

        if (this->is_antiprompt) {
          break;
        }
      }
    }

    if (this->embd.back() == llama_token_eos()) {
      break;
    }

    // display text
    if (!input_noecho && publish) {
      for (auto id : this->embd) {
        std::string aux_s = llama_token_to_str(this->ctx, id);

        result.append(aux_s);
        RCLCPP_INFO(this->get_logger(), "Generating text...");

        std_msgs::msg::String msg;
        msg.data = aux_s;
        this->text_pub->publish(msg);
      }
    }

    // respect the maximum number of tokens
    if (this->n_remain <= 0 && this->n_predict != -1) {
      this->n_remain = this->n_predict;
      break;
    }
  }

  return result;
}

void sigint_handler(int signo) {
  if (signo == SIGINT) {
    _exit(130);
  }
}

int main(int argc, char *argv[]) {

  struct sigaction sigint_action;
  sigint_action.sa_handler = sigint_handler;
  sigemptyset(&sigint_action.sa_mask);
  sigint_action.sa_flags = 0;
  sigaction(SIGINT, &sigint_action, NULL);

  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LlamaNode>());
  rclcpp::shutdown();
  return 0;
}
