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

  std::string model;
  std::string lora_adapter;
  std::string lora_base;
  bool numa;
  std::string prefix;
  std::string suffix;
  std::string stop;

  std::vector<double> tensor_split;

  std::string prompt;
  std::string file_path;

  auto context_params = llama_context_default_params();
  auto eval_params = llama_eval_default_params();

  // node params from llama.cpp common.h
  this->declare_parameters<int32_t>("", {
                                            {"seed", -1},
                                            {"n_threads", 1},
                                            {"n_predict", 128},
                                            {"n_ctx", 512},
                                            {"n_batch", 512},
                                            {"n_keep", -1},
                                            {"n_gpu_layers", 0},
                                            {"main_gpu", 0},
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
                                          {"rope_freq_base", 10000.0f},
                                          {"rope_freq_scale", 1.0f},
                                      });
  this->declare_parameter<std::vector<double>>("tensor_split",
                                               std::vector<double>({0.0}));
  this->declare_parameters<bool>("", {
                                         {"memory_f16", true},
                                         {"use_mmap", true},
                                         {"use_mlock", false},
                                         {"embedding", true},
                                         {"low_vram", false},
                                         {"numa", false},
                                     });

  this->get_parameter("seed", context_params.seed);
  this->get_parameter("n_ctx", context_params.n_ctx);
  this->get_parameter("memory_f16", context_params.f16_kv);
  this->get_parameter("use_mmap", context_params.use_mmap);
  this->get_parameter("use_mlock", context_params.use_mlock);
  this->get_parameter("embedding", context_params.embedding);

  this->get_parameter("n_gpu_layers", context_params.n_gpu_layers);
  this->get_parameter("main_gpu", context_params.main_gpu);
  this->get_parameter("tensor_split", tensor_split);
  this->get_parameter("low_vram", context_params.low_vram);

  this->get_parameter("rope_freq_scale", context_params.rope_freq_scale);
  this->get_parameter("rope_freq_base", context_params.rope_freq_base);

  this->get_parameter("n_threads", eval_params.n_threads);
  this->get_parameter("n_predict", eval_params.n_predict);
  this->get_parameter("n_keep", eval_params.n_keep);
  this->get_parameter("n_batch", eval_params.n_batch);

  this->get_parameter("model", model);
  this->get_parameter("lora_adapter", lora_adapter);
  this->get_parameter("lora_base", lora_base);
  this->get_parameter("numa", numa);

  this->get_parameter("prefix", prefix);
  this->get_parameter("suffix", suffix);
  this->get_parameter("stop", stop);

  this->get_parameter("prompt", prompt);
  this->get_parameter("file", file_path);

  // parse tensor_split
  for (size_t i = 0; i < LLAMA_MAX_DEVICES; ++i) {
    if (i < tensor_split.size()) {
      context_params.tensor_split[i] = (float)tensor_split[i];
    } else {
      context_params.tensor_split[i] = 0.0f;
    }
  }

  // load llama
  this->llama =
      std::make_shared<Llama>(context_params, eval_params, model, lora_adapter,
                              lora_base, numa, prefix, suffix, stop);

  // initial prompt
  if (!file_path.empty()) {
    std::ifstream file(file_path.c_str());
    if (!file) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open file %s",
                   file_path.c_str());
    }
    std::copy(std::istreambuf_iterator<char>(file),
              std::istreambuf_iterator<char>(), back_inserter(prompt));
  }
  this->llama->generate_response(prompt, false, llama_sampling_default_params(),
                                 nullptr);

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

void LlamaNode::tokenize_service_callback(
    const std::shared_ptr<llama_msgs::srv::Tokenize::Request> request,
    std::shared_ptr<llama_msgs::srv::Tokenize::Response> response) {

  response->tokens = this->llama->tokenize(request->prompt, false);
}

void LlamaNode::generate_embeddings_service_callback(
    const std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Request> request,
    std::shared_ptr<llama_msgs::srv::GenerateEmbeddings::Response> response) {

  if (this->llama->embedding) {
    response->embeddings = this->llama->create_embeddings(request->prompt);
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

  RCLCPP_INFO(this->get_logger(), "Prompt received: %s", prompt.c_str());

  // reset llama
  if (reset) {
    this->llama->reset();
  }

  // prepare sampling params
  auto sampling_params = llama_sampling_default_params();
  sampling_params.temp = sampling_config.temp;
  sampling_params.top_k = sampling_config.top_k;
  sampling_params.top_p = sampling_config.top_p;
  sampling_params.tfs_z = sampling_config.tfs_z;
  sampling_params.typical_p = sampling_config.typical_p;
  sampling_params.repeat_last_n = sampling_config.repeat_last_n;
  sampling_params.repeat_penalty = sampling_config.repeat_penalty;
  sampling_params.presence_penalty = sampling_config.presence_penalty;
  sampling_params.frequency_penalty = sampling_config.frequency_penalty;
  sampling_params.mirostat = sampling_config.mirostat;
  sampling_params.mirostat_eta = sampling_config.mirostat_eta;
  sampling_params.mirostat_tau = sampling_config.mirostat_tau;
  sampling_params.penalize_nl = sampling_config.penalize_nl;
  sampling_params.n_probs = sampling_config.n_probs;

  // call llama
  result->response = this->llama->generate_response(
      prompt, true, sampling_params,
      std::bind(&LlamaNode::send_text, this, _1));

  if (rclcpp::ok()) {

    if (this->goal_handle_->is_canceling()) {
      this->goal_handle_->canceled(result);
    } else {
      this->goal_handle_->succeed(result);
    }

    this->goal_handle_ = nullptr;
  }
}

void LlamaNode::send_text(const std::string &text) {
  if (this->goal_handle_ != nullptr) {
    auto feedback = std::make_shared<GenerateResponse::Feedback>();
    feedback->text = text;
    this->goal_handle_->publish_feedback(feedback);
  }
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
