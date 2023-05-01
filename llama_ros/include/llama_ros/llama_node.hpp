#ifndef LLAMA_NODE_HPP
#define LLAMA_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>

#include <memory>
#include <string>

#include "llama.h"
#include "llama_msgs/action/gpt.hpp"
#include "llama_ros/llama.hpp"

namespace llama_ros {

class LlamaNode : public rclcpp::Node {

  using GPT = llama_msgs::action::GPT;
  using GoalHandleGPT = rclcpp_action::ServerGoalHandle<GPT>;

public:
  LlamaNode();

private:
  std::shared_ptr<Llama> llama;

  // ros2
  rclcpp_action::Server<GPT>::SharedPtr gpt_action_server_;
  GPT::Goal current_goal_;
  std::shared_ptr<GoalHandleGPT> goal_handle_;
  std::mutex handle_accepted_mtx_;

  // methods
  void process_initial_prompt(std::string prompt);

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
