
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

  int32_t n_threads;
  int32_t n_predict;
  int32_t repeat_last_n;
  int32_t n_batch;
  int32_t n_keep;

  float temp;
  int32_t top_k;
  float top_p;
  float tfs_z;
  float typical_p;
  float repeat_penalty;
  float presence_penalty;
  float frequency_penalty;
  int32_t mirostat;
  float mirostat_tau;
  float mirostat_eta;
  bool penalize_nl;

  std::string prompt;
  std::string file_path;
  std::string model;
  std::string lora_adapter;
  std::string lora_base;
  std::string prefix;
  std::string suffix;
  std::string stop;

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

  this->get_parameter("n_threads", n_threads);
  this->get_parameter("n_predict", n_predict);
  this->get_parameter("n_keep", n_keep);
  this->get_parameter("n_batch", n_batch);
  this->get_parameter("repeat_last_n", repeat_last_n);

  this->get_parameter("temp", temp);
  this->get_parameter("top_k", top_k);
  this->get_parameter("top_p", top_p);
  this->get_parameter("tfs_z", tfs_z);
  this->get_parameter("typical_p", typical_p);
  this->get_parameter("presence_penalty", presence_penalty);
  this->get_parameter("frequency_penalty", frequency_penalty);
  this->get_parameter("mirostat", mirostat);
  this->get_parameter("mirostat_tau", mirostat_tau);
  this->get_parameter("mirostat_eta", mirostat_eta);
  this->get_parameter("penalize_nl", penalize_nl);
  this->get_parameter("repeat_penalty", repeat_penalty);

  this->get_parameter("model", model);
  this->get_parameter("lora_adapter", lora_adapter);
  this->get_parameter("lora_base", lora_base);
  this->get_parameter("prompt", prompt);
  this->get_parameter("file", file_path);
  this->get_parameter("prefix", prefix);
  this->get_parameter("suffix", suffix);
  this->get_parameter("stop", stop);

  // load llama
  this->llama = std::make_shared<Llama>(
      lparams, n_threads, n_predict, repeat_last_n, n_batch, n_keep, temp,
      top_k, top_p, tfs_z, typical_p, repeat_penalty, presence_penalty,
      frequency_penalty, mirostat, mirostat_tau, mirostat_eta, penalize_nl,
      model, lora_adapter, lora_base, prefix, suffix, stop);

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
  this->llama->generate_response(prompt, false, nullptr);

  // gpt action
  this->goal_handle_ = nullptr;
  this->gpt_action_server_ = rclcpp_action::create_server<GPT>(
      this, "gpt", std::bind(&LlamaNode::handle_goal, this, _1, _2),
      std::bind(&LlamaNode::handle_cancel, this, _1),
      std::bind(&LlamaNode::handle_accepted, this, _1));

  RCLCPP_INFO(this->get_logger(), "Llama Node started");
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
  this->llama->cancel();
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
    this->llama->reset();
  }

  if (embedding) {
    result->embeddings = this->llama->create_embeddings(prompt);

  } else {
    result->response = this->llama->generate_response(
        prompt, true, std::bind(&LlamaNode::send_text, this, _1));
  }

  if (rclcpp::ok()) {

    if (embedding) {
      if (this->llama->embedding) {
        this->goal_handle_->succeed(result);
      } else {
        this->goal_handle_->abort(result);
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
  if (this->goal_handle_ != nullptr) {
    RCLCPP_INFO(this->get_logger(), "Generating text...");
    auto feedback = std::make_shared<GPT::Feedback>();
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
