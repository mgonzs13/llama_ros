
#ifndef LLAMA_NODE_H
#define LLAMA_NODE_H

#include <memory>
#include <random>
#include <rclcpp/rclcpp.hpp>

#include "llama.h"
#include "llama_msgs/srv/gpt.hpp"

class LlamaNode : public rclcpp::Node {
public:
  LlamaNode();
  ~LlamaNode();

  void gpt_cb(const std::shared_ptr<llama_msgs::srv::GPT::Request> request,
              std::shared_ptr<llama_msgs::srv::GPT::Response> response);

protected:
  llama_context *ctx;

private:
  int32_t seed; // RNG seed
  int32_t n_threads;
  int32_t n_predict;     // new tokens to predict
  int32_t repeat_last_n; // last n tokens to penalize
  int32_t
      n_parts;   // amount of model parts (-1 = determine from model dimensions)
  int32_t n_ctx; // context size
  int32_t n_batch; // batch size for prompt processing
  int32_t n_keep;  // number of tokens to keep from initial prompt

  // sampling parameters
  int32_t top_k;
  float top_p;
  float temp;
  float repeat_penalty;

  std::string model;        // model path
  std::string input_prefix; // string to prefix user inputs with

  std::vector<std::string>
      antiprompt; // string upon seeing which more user input is prompted

  bool memory_f16;  // use f16 instead of f32 for memory kv
  bool interactive; // interactive mode

  bool interactive_start; // wait for user input immediately

  bool instruct;       // instruction mode (used for Alpaca models)
  bool ignore_eos;     // do not stop generating after eos
  bool use_mlock;      // use mlock to keep model in memory
  bool verbose_prompt; // print prompt tokens before generation

  bool is_interacting = false;
  std::vector<llama_token> embd_inp;
  std::vector<llama_token> inp_pfx;
  std::vector<llama_token> inp_sfx;
  std::vector<llama_token> llama_token_newline;

  bool is_antiprompt = false;
  bool input_noecho = false;

  int n_past = 0;
  int n_remain = this->n_predict;
  int n_consumed = 0;

  rclcpp::Service<llama_msgs::srv::GPT>::SharedPtr gpt_service;

  std::vector<llama_token> llama_node_tokenize(struct llama_context *ctx,
                                               const std::string &text,
                                               bool add_bos);
  void process_initial_prompt(std::string prompt);
  std::string process_prompt();
};

#endif
