#ifndef LLAMA_NODE_HPP
#define LLAMA_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include "llama.h"
#include "llama_msgs/action/gpt.hpp"

namespace llama_ros {

class LlamaNode : public rclcpp::Node {

  using GPT = llama_msgs::action::GPT;
  using GoalHandleGPT = rclcpp_action::ServerGoalHandle<GPT>;

public:
  LlamaNode();
  ~LlamaNode();

  std::string detokenize(const std::vector<llama_token> &tokens);
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
  bool embedding;

  // sampling parameters
  float temp;
  int32_t top_k;
  float top_p;
  float tfs_z;
  float typical_p;
  float repeat_penalty;
  float presence_penalty;
  float frequency_penalty;
  int mirostat;
  float mirostat_tau;
  float mirostat_eta;
  bool penalize_nl;

  // prefix, suffix, stop
  std::string stop;
  std::vector<llama_token> inp_pfx;
  std::vector<llama_token> inp_sfx;

  // aux
  std::vector<llama_token> last_n_tokens;
  std::vector<llama_token> prompt_tokens;
  std::vector<llama_token> batch_tokens;

  bool is_antiprompt;
  bool input_noecho;
  int32_t n_past;
  int32_t n_remain;
  int32_t n_consumed;

  // ros2
  rclcpp_action::Server<GPT>::SharedPtr gpt_action_server_;
  GPT::Goal current_goal_;
  std::shared_ptr<GoalHandleGPT> goal_handle_;
  std::mutex handle_accepted_mtx_;

  // methods
  void process_initial_prompt(std::string prompt);
  std::string generate();
  std::vector<float> create_embeddings(std::string prompt);

  rclcpp_action::GoalResponse
  handle_goal(const rclcpp_action::GoalUUID &uuid,
              std::shared_ptr<const GPT::Goal> goal);
  rclcpp_action::CancelResponse
  handle_cancel(const std::shared_ptr<GoalHandleGPT> goal_handle);
  void handle_accepted(const std::shared_ptr<GoalHandleGPT> goal_handle);

  void execute(const std::shared_ptr<GoalHandleGPT> goal_handle);
  void send_text(const std::string &text);
};

} // namespace llama_ros

#endif
