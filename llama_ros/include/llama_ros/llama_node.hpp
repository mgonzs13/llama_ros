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

  void gpt_cb(const std::shared_ptr<llama_msgs::srv::GPT::Request> request,
              std::shared_ptr<llama_msgs::srv::GPT::Response> response);

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

  std::string input_prefix; // string to prefix user inputs with

  std::vector<std::string>
      antiprompt; // string upon seeing which more user input is prompted
  bool instruct;  // instruction mode (used for Alpaca models)

  std::vector<llama_token> embd_inp;
  std::vector<llama_token> inp_pfx;
  std::vector<llama_token> inp_sfx;
  std::vector<llama_token> llama_token_newline;
  std::vector<llama_token> last_n_tokens;
  std::vector<llama_token> embd;

  bool is_antiprompt;
  bool input_noecho;

  int n_past;
  int n_remain;
  int n_consumed;

  rclcpp::Service<llama_msgs::srv::GPT>::SharedPtr gpt_service;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr text_pub;

  std::vector<llama_token> tokenize(const std::string &text, bool add_bos);
  void process_initial_prompt(std::string prompt);
  std::string process_prompt(bool publish);
};

} // namespace llama_ros

#endif
