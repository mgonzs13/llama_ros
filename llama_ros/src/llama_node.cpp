
#include <fstream>
#include <memory>
#include <signal.h>
#include <string>
#include <unistd.h>
#include <vector>

#include "llama.h"
#include "llama_ros/llama_node.hpp"

using namespace llama_ros;
using std::placeholders::_1;
using std::placeholders::_2;

LlamaNode::LlamaNode() : rclcpp::Node("llama_node") {

  std::string prompt;
  std::string file_path;
  std::string model;
  std::string lora_adapter;
  std::string lora_base;
  std::string prefix;
  std::string suffix;
  auto lparams = llama_context_default_params();

  // node params from llama.cpp common.h
  this->declare_parameters<int32_t>("", {
                                            {"seed", -1},
                                            {"n_threads", 1},
                                            {"n_predict", 128},
                                            {"repeat_last_n", 64},
                                            {"n_parts", -1},
                                            {"n_ctx", 512},
                                            {"n_batch", 512},
                                            {"n_keep", -1},
                                            {"top_k", 40},
                                            {"mirostat", 0},
                                        });
  this->declare_parameters<std::string>("", {
                                                {"model", ""},
                                                {"lora_adapter", ""},
                                                {"lora_base", ""},
                                                {"prompt", ""},
                                                {"file", ""},
                                                {"prefix", ""},
                                                {"suffix", ""},
                                                {"stop", ""},
                                            });
  this->declare_parameters<float>("", {
                                          {"temp", 0.80f},
                                          {"top_p", 0.95f},
                                          {"tfs_z", 1.00f},
                                          {"typical_p", 1.00f},
                                          {"presence_penalty", 0.00f},
                                          {"frequency_penalty", 0.00f},
                                          {"mirostat_tau", 5.10f},
                                          {"mirostat_eta", 0.10f},
                                          {"repeat_penalty", 1.10f},
                                      });
  this->declare_parameters<bool>("", {
                                         {"memory_f16", true},
                                         {"use_mmap", true},
                                         {"use_mlock", false},
                                         {"embedding", true},
                                         {"penalize_nl", true},
                                     });

  this->get_parameter("seed", lparams.seed);
  this->get_parameter("n_parts", lparams.n_parts);
  this->get_parameter("n_ctx", lparams.n_ctx);
  this->get_parameter("memory_f16", lparams.f16_kv);
  this->get_parameter("use_mmap", lparams.use_mmap);
  this->get_parameter("use_mlock", lparams.use_mlock);
  this->get_parameter("embedding", lparams.embedding);
  this->embedding = lparams.embedding;

  this->get_parameter("n_threads", this->n_threads);
  this->get_parameter("n_predict", this->n_predict);
  this->get_parameter("n_keep", this->n_keep);
  this->get_parameter("n_batch", this->n_batch);
  this->get_parameter("repeat_last_n", this->repeat_last_n);
  this->repeat_last_n =
      this->repeat_last_n < 0 ? lparams.n_ctx : this->repeat_last_n;

  this->get_parameter("temp", this->temp);
  this->get_parameter("top_k", this->top_k);
  this->get_parameter("top_p", this->top_p);
  this->get_parameter("tfs_z", this->tfs_z);
  this->get_parameter("typical_p", this->typical_p);
  this->get_parameter("presence_penalty", this->presence_penalty);
  this->get_parameter("frequency_penalty", this->frequency_penalty);
  this->get_parameter("mirostat", this->mirostat);
  this->get_parameter("mirostat_tau", this->mirostat_tau);
  this->get_parameter("mirostat_eta", this->mirostat_eta);
  this->get_parameter("penalize_nl", this->penalize_nl);
  this->get_parameter("repeat_penalty", this->repeat_penalty);

  this->get_parameter("model", model);
  this->get_parameter("lora_adapter", lora_adapter);
  this->get_parameter("lora_base", lora_base);
  this->get_parameter("prompt", prompt);
  this->get_parameter("file", file_path);
  this->get_parameter("prefix", prefix);
  this->get_parameter("suffix", suffix);
  this->get_parameter("stop", this->stop);

  // load prompt from file
  if (!file_path.empty()) {
    std::ifstream file(file_path.c_str());
    if (!file) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open file %s",
                   file_path.c_str());
    }
    std::copy(std::istreambuf_iterator<char>(file),
              std::istreambuf_iterator<char>(), back_inserter(prompt));
  }

  // when using lora, mmap is disable
  if (!lora_adapter.empty()) {
    lparams.use_mmap = false;
  }

  // load the model
  this->ctx = llama_init_from_file(model.c_str(), lparams);
  this->n_ctx = llama_n_ctx(this->ctx);

  if (this->n_ctx > 2048) {
    RCLCPP_WARN(this->get_logger(),
                "Model does not support context sizes greater than 2048 tokens "
                "(%d specified); expect poor results",
                this->n_ctx);
  }

  if (this->ctx == NULL) {
    RCLCPP_ERROR(this->get_logger(), "Failed to load model '%s'",
                 model.c_str());
  }

  if (!lora_adapter.empty()) {
    if (llama_apply_lora_from_file(ctx, lora_adapter.c_str(),
                                   lora_base.empty() ? NULL : lora_base.c_str(),
                                   this->n_threads)) {
      RCLCPP_ERROR(this->get_logger(), "Failed to apply lora adapter");
    }
  }

  // show system information
  RCLCPP_INFO(this->get_logger(), "System_info: n_threads = %d / %d | %s",
              this->n_threads, std::thread::hardware_concurrency(),
              llama_print_system_info());

  // prefix & suffix
  this->inp_pfx = this->tokenize(prefix, true);
  this->inp_sfx = this->tokenize(suffix, false);

  // number of tokens to keep when resetting context
  if (this->n_keep == -1) {
    this->n_keep = (int)this->prompt_tokens.size();
  }

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

  // gpt action
  this->goal_handle_ = nullptr;
  this->gpt_action_server_ = rclcpp_action::create_server<GPT>(
      this, "gpt", std::bind(&LlamaNode::handle_goal, this, _1, _2),
      std::bind(&LlamaNode::handle_cancel, this, _1),
      std::bind(&LlamaNode::handle_accepted, this, _1));

  RCLCPP_INFO(this->get_logger(), "Llama Node started");
}

LlamaNode::~LlamaNode() { llama_free(this->ctx); }

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

std::string LlamaNode::detokenize(const std::vector<llama_token> &tokens) {
  std::string output = "";
  for (llama_token token : tokens) {
    output += llama_token_to_str(this->ctx, token);
  }
  return output;
}

void LlamaNode::process_initial_prompt(std::string prompt) {

  // Add a space in front of the first character to match OG llama tokenizer
  // behavior
  prompt.insert(0, 1, ' ');

  // tokenize the prompt
  this->prompt_tokens = this->tokenize(prompt, true);

  if ((int)this->prompt_tokens.size() > this->n_ctx - 4) {
    RCLCPP_WARN(this->get_logger(), "Prompt is too long (%d tokens, max %d)",
                (int)this->prompt_tokens.size(), this->n_ctx - 4);
  }

  if (prompt.length() > 1) {
    this->generate();
  }
}

std::string LlamaNode::generate() {

  bool input_noecho = true;
  std::string result;
  std::string stopping_text;
  std::string aux;

  while (this->n_remain != 0) {

    // predict
    if (this->batch_tokens.size() > 0) {
      // infinite text generation via context swapping
      // if we run out of context:
      // - take the n_keep first tokens from the original prompt (via n_past)
      // - take half of the last (n_ctx - n_keep) tokens and recompute the
      // logits in a batch
      if (this->n_past + (int)this->batch_tokens.size() > this->n_ctx) {

        const int n_left = this->n_past - this->n_keep;
        this->n_past = this->n_keep;

        // insert n_left/2 tokens at the start of batch_tokens
        // from last_n_tokens
        this->batch_tokens.insert(this->batch_tokens.begin(),
                                  this->last_n_tokens.begin() + this->n_ctx -
                                      n_left / 2 - this->batch_tokens.size(),
                                  this->last_n_tokens.end() -
                                      this->batch_tokens.size());
      }

      RCLCPP_INFO(this->get_logger(), "EVAL %d",
                  (int)this->batch_tokens.size());
      if (llama_eval(this->ctx, this->batch_tokens.data(),
                     this->batch_tokens.size(), this->n_past,
                     this->n_threads)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to eval");
      }
    }

    this->n_past += this->batch_tokens.size();
    this->batch_tokens.clear();

    if ((int)this->prompt_tokens.size() <= this->n_consumed) {
      // out of user input, sample next token

      llama_token id = 0;

      {
        auto logits = llama_get_logits(this->ctx);
        auto n_vocab = llama_n_vocab(this->ctx);

        std::vector<llama_token_data> candidates;
        candidates.reserve(n_vocab);
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
          candidates.emplace_back(
              llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {candidates.data(),
                                               candidates.size(), false};

        // Apply penalties
        float nl_logit = logits[llama_token_nl()];
        auto last_n_repeat = std::min(
            std::min((int)this->last_n_tokens.size(), this->repeat_last_n),
            this->n_ctx);

        llama_sample_repetition_penalty(this->ctx, &candidates_p,
                                        this->last_n_tokens.data() +
                                            this->last_n_tokens.size() -
                                            last_n_repeat,
                                        last_n_repeat, this->repeat_penalty);
        llama_sample_frequency_and_presence_penalties(
            this->ctx, &candidates_p,
            this->last_n_tokens.data() + this->last_n_tokens.size() -
                last_n_repeat,
            last_n_repeat, this->frequency_penalty, this->presence_penalty);
        if (!this->penalize_nl) {
          logits[llama_token_nl()] = nl_logit;
        }

        if (this->temp <= 0) {
          // Greedy sampling
          id = llama_sample_token_greedy(this->ctx, &candidates_p);
        } else {
          if (this->mirostat == 1) {
            static float mirostat_mu = 2.0f * this->mirostat_tau;
            const int mirostat_m = 100;
            llama_sample_temperature(this->ctx, &candidates_p, this->temp);
            id = llama_sample_token_mirostat(
                this->ctx, &candidates_p, this->mirostat_tau,
                this->mirostat_eta, mirostat_m, &mirostat_mu);
          } else if (this->mirostat == 2) {
            static float mirostat_mu = 2.0f * this->mirostat_tau;
            llama_sample_temperature(ctx, &candidates_p, temp);
            id = llama_sample_token_mirostat_v2(
                this->ctx, &candidates_p, this->mirostat_tau,
                this->mirostat_eta, &mirostat_mu);
          } else {
            // Temperature sampling
            llama_sample_top_k(this->ctx, &candidates_p, this->top_k);
            llama_sample_tail_free(this->ctx, &candidates_p, this->tfs_z);
            llama_sample_typical(this->ctx, &candidates_p, this->typical_p);
            llama_sample_top_p(this->ctx, &candidates_p, this->top_p);
            llama_sample_temperature(this->ctx, &candidates_p, this->temp);
            id = llama_sample_token(this->ctx, &candidates_p);
          }
        }

        this->last_n_tokens.erase(last_n_tokens.begin());
        this->last_n_tokens.push_back(id);
      }

      // add it to the context
      this->batch_tokens.push_back(id);

      // echo this to console
      input_noecho = false;

      // decrement remaining sampling budget
      --this->n_remain;

    } else {
      // some user input remains from prompt, forward it to processing
      while ((int)this->prompt_tokens.size() > this->n_consumed) {
        this->batch_tokens.push_back(this->prompt_tokens[this->n_consumed]);
        this->last_n_tokens.erase(this->last_n_tokens.begin());
        this->last_n_tokens.push_back(this->prompt_tokens[this->n_consumed]);
        ++this->n_consumed;
        if ((int)this->batch_tokens.size() >= this->n_batch) {
          break;
        }
      }
    }

    // when not currently processing queued
    // inputs check if we should end
    if ((int)this->prompt_tokens.size() <= this->n_consumed) {

      // check if stop appears at the end of the output
      std::string last_output = this->detokenize(this->last_n_tokens);
      this->is_antiprompt = false;

      if (this->stop.size() &&
          last_output.find(this->stop.c_str(),
                           last_output.length() - this->stop.length(),
                           this->stop.length()) != std::string::npos) {
        this->is_antiprompt = true;
        break;
      }
    }

    if (this->batch_tokens.back() == llama_token_eos()) {
      break;
    }

    if (this->goal_handle_ != nullptr) {
      if (this->goal_handle_->is_canceling()) {
        RCLCPP_INFO(this->get_logger(), "Action Canceled");
        break;
      }
    }

    // check if stop tokens appears at the end of the output
    aux = this->detokenize(this->batch_tokens);
    if (((int)this->prompt_tokens.size() <= this->n_consumed) &&
        this->stop.size() &&
        this->stop.find(aux.c_str(), stopping_text.size(), aux.length()) !=
            std::string::npos) {

      // remove and send chars before stop
      if (!stopping_text.size()) {
        for (int i = 0; i < (int)aux.size(); i++) {
          if (aux.at(0) != this->stop[i]) {

            if (!input_noecho && this->goal_handle_ != nullptr) {
              this->send_text(aux.substr(0, 1));
              result.append(aux.substr(0, 1));
            }

            aux.erase(aux.begin());

          } else {
            break;
          }
        }
      }

      stopping_text.append(aux);

      if (stopping_text.size() == this->stop.size()) {
        this->is_antiprompt = true;
        break;
      }

    } else {

      // send text
      if (!input_noecho && this->goal_handle_ != nullptr) {
        std::string text = aux;

        if (stopping_text.size()) {
          text = stopping_text + text;
          stopping_text.clear();
        }

        this->send_text(text);
        result.append(text);
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

std::vector<float> LlamaNode::create_embeddings(std::string prompt) {

  if (!this->embedding) {
    RCLCPP_ERROR(
        this->get_logger(),
        "Llama must be created with embedding=true to create embeddings");
    this->goal_handle_->abort(std::make_shared<GPT::Result>());
    return {};
  }

  prompt.insert(0, 1, ' ');
  auto tokens = this->tokenize(prompt, true);

  if (llama_eval(ctx, tokens.data(), tokens.size(), 0, this->n_threads)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to eval");
  }

  const int n_embd = llama_n_embd(ctx);
  const auto embeddings = llama_get_embeddings(ctx);
  std::vector<float> embeddings_list;

  for (int i = 0; i < n_embd; i++) {
    embeddings_list.push_back(embeddings[i]);
  }

  return embeddings_list;
}

rclcpp_action::GoalResponse
LlamaNode::handle_goal(const rclcpp_action::GoalUUID &uuid,
                       std::shared_ptr<const GPT::Goal> goal) {
  (void)uuid;
  (void)goal;

  if (this->goal_handle_ != nullptr && this->goal_handle_->is_active()) {
    return rclcpp_action::GoalResponse::REJECT;
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse
LlamaNode::handle_cancel(const std::shared_ptr<GoalHandleGPT> goal_handle) {
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "Received request to cancel");
  return rclcpp_action::CancelResponse::ACCEPT;
}

void LlamaNode::handle_accepted(
    const std::shared_ptr<GoalHandleGPT> goal_handle) {
  this->goal_handle_ = goal_handle;
  std::thread{std::bind(&LlamaNode::execute, this, _1), goal_handle}.detach();
}

void LlamaNode::execute(const std::shared_ptr<GoalHandleGPT> goal_handle) {

  auto result = std::make_shared<GPT::Result>();
  std::string prompt = goal_handle->get_goal()->prompt;
  bool embedding = goal_handle->get_goal()->embedding;
  bool reset = goal_handle->get_goal()->reset;
  this->goal_handle_ = goal_handle;

  RCLCPP_INFO(this->get_logger(), "Prompt received: %s", prompt.c_str());

  if (reset) {
    this->last_n_tokens = std::vector<llama_token>(this->n_ctx);
    std::fill(this->last_n_tokens.begin(), this->last_n_tokens.end(), 0);

    this->n_past = 0;
    this->is_antiprompt = false;
    this->n_remain = this->n_predict;
    this->n_consumed = 0;

    this->prompt_tokens.clear();
    this->batch_tokens.clear();
  }

  if (prompt.length() > 0) {

    if (embedding) {

      result->embeddings = this->create_embeddings(prompt);

    } else {

      if (!this->prompt_tokens.size()) {
        prompt.insert(0, 1, ' ');
      }
      auto line_inp = this->tokenize(prompt, false);

      int prompt_size = this->prompt_tokens.size() + this->inp_pfx.size() +
                        line_inp.size() + this->inp_sfx.size();
      if (prompt_size > this->n_ctx - 4) {
        RCLCPP_WARN(this->get_logger(),
                    "Prompt is too long (%d tokens, max %d)", prompt_size,
                    this->n_ctx - 4);
      }

      // insert prefix
      if (this->inp_pfx.size() && !this->is_antiprompt) {
        this->n_consumed = this->prompt_tokens.size();
        this->prompt_tokens.insert(this->prompt_tokens.end(),
                                   this->inp_pfx.begin(), this->inp_pfx.end());
      }

      this->prompt_tokens.insert(this->prompt_tokens.end(), line_inp.begin(),
                                 line_inp.end());

      // insert suffix
      if (this->inp_sfx.size()) {
        this->prompt_tokens.insert(this->prompt_tokens.end(),
                                   this->inp_sfx.begin(), this->inp_sfx.end());
      }

      this->n_remain -= line_inp.size();

      result->response = this->generate();
    }
  }

  if (rclcpp::ok()) {

    if (embedding) {
      if (this->embedding) {
        this->goal_handle_->succeed(result);
      }

    } else {
      if (this->goal_handle_->is_canceling()) {
        this->goal_handle_->canceled(result);
      } else {
        this->goal_handle_->succeed(result);
      }
    }

    this->goal_handle_ = nullptr;
  }
}

void LlamaNode::send_text(const std::string &text) {
  RCLCPP_INFO(this->get_logger(), "Generating text...");
  auto feedback = std::make_shared<GPT::Feedback>();
  feedback->text = text;
  this->goal_handle_->publish_feedback(feedback);
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
