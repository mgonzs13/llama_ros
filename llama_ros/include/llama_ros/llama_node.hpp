#ifndef LLAMA_NODE_HPP
#define LLAMA_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include "llama.h"
#include "llama_msgs/srv/gpt.hpp"

namespace llama_ros {

class LlamaNode : public rclcpp::Node {
public:
  LlamaNode();
  ~LlamaNode();

  std::string detokenize(std::vector<llama_token> tokens);
  std::vector<llama_token> tokenize(const std::string &text, bool add_bos);

protected:
  llama_context *ctx;

private:
  int32_t n_threads;
  int32_t n_predict;     // new tokens to predict
  int32_t repeat_last_n; // last n tokens to penalize
  int32_t n_ctx;         // context size
  int32_t n_batch;       // batch size for prompt processing
  int32_t n_keep;        // number of tokens to keep from initial prompt

  // sampling parameters
  int32_t top_k;
  float top_p;
  float temp;
  float repeat_penalty;

  // prefix, suffix, stop
  std::string stop;
  std::vector<llama_token> inp_pfx;
  std::vector<llama_token> inp_sfx;

  // aux
  std::vector<llama_token> last_n_tokens;
  std::vector<llama_token> embd_inp;
  std::vector<llama_token> embd;

  bool is_antiprompt;
  bool input_noecho;
  int n_past;
  int n_remain;
  int n_consumed;

  // ros2
  rclcpp::Service<llama_msgs::srv::GPT>::SharedPtr gpt_service;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr text_pub;

  // methods
  void process_initial_prompt(std::string prompt);
  std::string generate(bool publish);
  void gpt_cb(const std::shared_ptr<llama_msgs::srv::GPT::Request> request,
              std::shared_ptr<llama_msgs::srv::GPT::Response> response);
};

} // namespace llama_ros

#endif
