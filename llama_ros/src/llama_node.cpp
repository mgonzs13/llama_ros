// MIT License

// Copyright (c) 2023  Miguel Ángel González Santamarta

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "common.h"
#include "llama.h"
#include "llama_msgs/msg/token_prob.hpp"
#include "llama_msgs/msg/token_prob_array.hpp"
#include "llama_ros/llama_node.hpp"

using namespace llama_ros;
using std::placeholders::_1;
using std::placeholders::_2;

LlamaNode::LlamaNode() : rclcpp::Node("llama_node") {

  // load llama
  struct gpt_params params;
  this->load_params(params);
  this->llama = std::make_shared<Llama>(params, debug);
  this->llama->generate_response(params.prompt, false, nullptr);

  // services
  this->tokenize_service_ = this->create_service<llama_msgs::srv::Tokenize>(
      "tokenize",
      std::bind(&LlamaNode::tokenize_service_callback, this, _1, _2));
  this->generate_embeddings_service_ =
      this->create_service<llama_msgs::srv::GenerateEmbeddings>(
          "generate_embeddings",
          std::bind(&LlamaNode::generate_embeddings_service_callback, this, _1,
                    _2));

  // generate reponse action server
  this->goal_handle_ = nullptr;
  this->generate_response_action_server_ =
      rclcpp_action::create_server<GenerateResponse>(
          this, "generate_response",
          std::bind(&LlamaNode::handle_goal, this, _1, _2),
          std::bind(&LlamaNode::handle_cancel, this, _1),
          std::bind(&LlamaNode::handle_accepted, this, _1));

  RCLCPP_INFO(this->get_logger(), "Llama Node started");
}

void LlamaNode::load_params(struct gpt_params &params) {

  std::string stop;
  std::string file_path;
  std::string lora_adapter;
  std::vector<double> tensor_split;

  // node params from llama.cpp common.h
  this->declare_parameters<int32_t>("", {
                                            {"seed", -1},
                                            {"n_ctx", 512},
                                            {"n_batch", 512},
                                            {"n_gpu_layers", 0},
                                            {"main_gpu", 0},
                                            {"n_threads", 1},
                                            {"n_predict", 128},
                                            {"n_keep", -1},
                                            {"n_parallel", 1},
                                            {"n_sequences", 1},
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
                                                {"cache_type_k", "f16"},
                                                {"cache_type_v", "f16"},
                                            });
  this->declare_parameters<float>("", {
                                          {"rope_freq_base", 0.0f},
                                          {"rope_freq_scale", 0.0f},
                                      });
  this->declare_parameter<std::vector<double>>("tensor_split",
                                               std::vector<double>({0.0}));
  this->declare_parameters<bool>("", {
                                         {"debug", true},
                                         {"mul_mat_q", true},
                                         {"embedding", true},
                                         {"logits_all", false},
                                         {"use_mmap", true},
                                         {"use_mlock", false},
                                         {"numa", false},
                                         {"cont_batching", false},
                                         {"dump_kv_cache", false},
                                         {"no_kv_offload", false},
                                     });

  this->get_parameter("seed", params.seed);
  this->get_parameter("n_ctx", params.n_ctx);
  this->get_parameter("n_batch", params.n_batch);

  this->get_parameter("n_gpu_layers", params.n_gpu_layers);
  this->get_parameter("main_gpu", params.main_gpu);
  this->get_parameter("tensor_split", tensor_split);

  this->get_parameter("rope_freq_scale", params.rope_freq_scale);
  this->get_parameter("rope_freq_base", params.rope_freq_base);

  this->get_parameter("mul_mat_q", params.mul_mat_q);
  this->get_parameter("embedding", params.embedding);
  this->get_parameter("logits_all", params.logits_all);
  this->get_parameter("use_mmap", params.use_mmap);
  this->get_parameter("use_mlock", params.use_mlock);

  this->get_parameter("dump_kv_cache", params.dump_kv_cache);
  this->get_parameter("no_kv_offload", params.no_kv_offload);
  this->get_parameter("cache_type_k", params.cache_type_k);
  this->get_parameter("cache_type_v", params.cache_type_v);

  this->get_parameter("n_threads", params.n_threads);
  this->get_parameter("n_predict", params.n_predict);
  this->get_parameter("n_keep", params.n_keep);
  this->get_parameter("n_batch", params.n_batch);

  this->get_parameter("model", params.model);
  this->get_parameter("lora_adapter", lora_adapter);
  this->get_parameter("lora_base", params.lora_base);
  this->get_parameter("numa", params.numa);

  this->get_parameter("n_parallel", params.n_parallel);
  this->get_parameter("n_sequences", params.n_sequences);
  this->get_parameter("cont_batching", params.cont_batching);

  this->get_parameter("prefix", params.input_prefix);
  this->get_parameter("suffix", params.input_suffix);
  this->get_parameter("stop", stop);

  this->get_parameter("prompt", params.prompt);
  this->get_parameter("file", file_path);
  this->get_parameter("debug", this->debug);

  // check threads number
  if (params.n_threads < 0) {
    params.n_threads = std::thread::hardware_concurrency();
  }

  // lora_adapter
  if (lora_adapter.size()) {
    params.lora_adapter.push_back({lora_adapter, 1.0f});
    params.use_mmap = false;
  }

  // stop is the antiprompt
  params.antiprompt.push_back(stop);

  // initial prompt
  if (!file_path.empty()) {
    std::ifstream file(file_path.c_str());
    if (!file) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open file %s",
                   file_path.c_str());
    }
    std::copy(std::istreambuf_iterator<char>(file),
              std::istreambuf_iterator<char>(), back_inserter(params.prompt));
  }

  // cublas
#ifdef GGML_USE_CUBLAS
  for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
    if (i < tensor_split.size()) {
      params.tensor_split[i] = tensor_split[i];
    } else {
      params.tensor_split[i] = 0.0f;
    }
  }

  params.mul_mat_q = false;
#endif
}

void LlamaNode::tokenize_service_callback(
    const std::shared_ptr<llama_msgs::srv::Tokenize::Request> request,
    std::shared_ptr<llama_msgs::srv::Tokenize::Response> response) {

  response->tokens = this->llama->tokenize(request->prompt, false);
}

void LlamaNode::generate_embeddings_service_callback(
    const std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Request> request,
    std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Response> response) {

  if (this->llama->is_embedding()) {
    response->embeddings = this->llama->generate_embeddings(request->prompt);
  }
}

rclcpp_action::GoalResponse
LlamaNode::handle_goal(const rclcpp_action::GoalUUID &uuid,
                       std::shared_ptr<const GenerateResponse::Goal> goal) {
  (void)uuid;
  (void)goal;

  if (this->goal_handle_ != nullptr && this->goal_handle_->is_active()) {
    return rclcpp_action::GoalResponse::REJECT;
  }

  return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse LlamaNode::handle_cancel(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) {
  (void)goal_handle;
  RCLCPP_INFO(this->get_logger(), "Received request to cancel");
  this->llama->cancel();
  return rclcpp_action::CancelResponse::ACCEPT;
}

void LlamaNode::handle_accepted(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) {
  this->goal_handle_ = goal_handle;
  std::thread{std::bind(&LlamaNode::execute, this, _1), goal_handle}.detach();
}

void LlamaNode::execute(
    const std::shared_ptr<GoalHandleGenerateResponse> goal_handle) {

  // get goal data
  std::string prompt = goal_handle->get_goal()->prompt;
  bool reset = goal_handle->get_goal()->reset;
  auto sampling_config = goal_handle->get_goal()->sampling_config;

  auto result = std::make_shared<GenerateResponse::Result>();
  this->goal_handle_ = goal_handle;

  if (this->debug) {
    RCLCPP_INFO(this->get_logger(), "Prompt received:\n%s", prompt.c_str());
  }

  // reset llama
  if (reset) {
    this->llama->reset();
  }

  // // prepare sampling params
  struct gpt_params &params = this->llama->get_params();
  params.ignore_eos = sampling_config.ignore_eos;
  params.sparams.temp = sampling_config.temp;
  params.sparams.top_k = sampling_config.top_k;
  params.sparams.top_p = sampling_config.top_p;
  params.sparams.tfs_z = sampling_config.tfs_z;
  params.sparams.typical_p = sampling_config.typical_p;
  params.sparams.penalty_last_n = sampling_config.penalty_last_n;
  params.sparams.penalty_repeat = sampling_config.penalty_repeat;
  params.sparams.penalty_present = sampling_config.penalty_present;
  params.sparams.penalty_freq = sampling_config.penalty_freq;
  params.sparams.mirostat = sampling_config.mirostat;
  params.sparams.mirostat_eta = sampling_config.mirostat_eta;
  params.sparams.mirostat_tau = sampling_config.mirostat_tau;
  params.sparams.penalize_nl = sampling_config.penalize_nl;
  params.sparams.n_probs = sampling_config.n_probs;
  params.sparams.grammar = sampling_config.grammar;

  // check penalty_last_n
  params.sparams.penalty_last_n = params.sparams.penalty_last_n < 0
                                      ? params.sparams.n_prev
                                      : params.sparams.penalty_last_n;

  // check top_k
  params.sparams.top_k = params.sparams.top_k <= 0 ? this->llama->get_n_vocab()
                                                   : params.sparams.top_k;

  // add logit bias
  for (auto logit_bias : sampling_config.logit_bias.data) {
    params.sparams.logit_bias[logit_bias.token] = logit_bias.bias;
  }

  // add llama_token_eos
  if (params.ignore_eos) {
    params.sparams.logit_bias[this->llama->get_token_eos()] = -INFINITY;
  }

  // call llama
  auto completion_results = this->llama->generate_response(
      prompt, true, std::bind(&LlamaNode::send_text, this, _1));

  for (auto completion : completion_results) {
    result->response.text.append(this->llama->detokenize({completion.token}));
    result->response.tokens.push_back(completion.token);

    llama_msgs::msg::TokenProbArray probs_msg;
    for (auto prob : completion.probs) {
      llama_msgs::msg::TokenProb aux;
      aux.token = prob.token;
      aux.probability = prob.probability;
      aux.token_text = this->llama->detokenize({prob.token});
      probs_msg.data.push_back(aux);
    }
    result->response.probs.push_back(probs_msg);
  }

  if (rclcpp::ok()) {

    if (this->goal_handle_->is_canceling()) {
      this->goal_handle_->canceled(result);
    } else {
      this->goal_handle_->succeed(result);
    }

    this->goal_handle_ = nullptr;
  }
}

void LlamaNode::send_text(const struct completion_output &completion) {

  if (this->goal_handle_ != nullptr) {
    auto feedback = std::make_shared<GenerateResponse::Feedback>();

    feedback->partial_response.text =
        this->llama->detokenize({completion.token});
    feedback->partial_response.token = completion.token;
    feedback->partial_response.probs.chosen_token = completion.token;

    for (auto prob : completion.probs) {
      llama_msgs::msg::TokenProb aux;
      aux.token = prob.token;
      aux.probability = prob.probability;
      aux.token_text = this->llama->detokenize({prob.token});
      feedback->partial_response.probs.data.push_back(aux);
    }

    this->goal_handle_->publish_feedback(feedback);
  }
}
