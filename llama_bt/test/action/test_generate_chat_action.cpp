
// MIT License
//
// Copyright (c) 2025 Alberto J. Tudela Roldán
// Copyright (c) 2025 Alejandro González Cantón
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "llama_msgs/msg/chat_choice.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <string>

#if defined(BTV3)
#include "behaviortree_cpp_v3/bt_factory.h"
#else
#include "behaviortree_cpp/bt_factory.h"
#endif

#include <ament_index_cpp/get_package_share_directory.hpp>

#include "llama_bt/action/generate_chat_completions_action.hpp"
#include "utils/test_action_server.hpp"

class GenerateResponseActionServer
    : public TestActionServer<llama_msgs::action::GenerateChatCompletions> {
public:
  GenerateResponseActionServer()
      : TestActionServer("generate_chat_completions") {}

protected:
  void execute(const typename std::shared_ptr<rclcpp_action::ServerGoalHandle<
                   llama_msgs::action::GenerateChatCompletions>>
                   goal_handle) override {
    llama_msgs::action::GenerateChatCompletions::Result::SharedPtr result =
        std::make_shared<llama_msgs::action::GenerateChatCompletions::Result>();
    llama_msgs::msg::ChatMessage chat_message;
    chat_message.content = "This is a test response";
    chat_message.role = "assistant";

    llama_msgs::msg::ChatChoice chat_choice;
    chat_choice.index = 0;
    chat_choice.message = chat_message;
    chat_choice.finish_reason = "stop";

    result->choices.push_back(chat_choice);

    bool return_success = getReturnSuccess();
    if (return_success) {
      goal_handle->succeed(result);
    } else {
      goal_handle->abort(result);
    }
  }
};

class GenerateChatActionTestFixture : public ::testing::Test {
public:
  void SetUp() override {
    rclcpp::init(0, nullptr);

    node_ = std::make_shared<rclcpp::Node>("generate_chat_test_fixture");
    factory_ = std::make_shared<BT::BehaviorTreeFactory>();

    config_ = new BT::NodeConfiguration();

    // Create the blackboard that will be shared by all of the nodes in the tree
    config_->blackboard = BT::Blackboard::create();
    // Put items on the blackboard
    config_->blackboard->set("node", node_);
    config_->blackboard->set<std::chrono::milliseconds>(
        "server_timeout", std::chrono::milliseconds(1000));
    config_->blackboard->set<std::chrono::milliseconds>(
        "bt_loop_duration", std::chrono::milliseconds(50));
    config_->blackboard->set<std::chrono::milliseconds>(
        "wait_for_service_timeout", std::chrono::milliseconds(1000));

    BT::NodeBuilder builder = [](const std::string &name,
                                 const BT::NodeConfiguration &config) {
      return std::make_unique<llama_bt::GenerateChatCompletionsAction>(
          name, "generate_chat_completions", config);
    };

    factory_->registerBuilder<llama_bt::GenerateChatCompletionsAction>(
        "GenerateChatCompletions", builder);

    server_ = std::make_shared<GenerateResponseActionServer>();
    server_thread_ = std::thread([this]() { rclcpp::spin(server_); });
  }

  void TearDown() override {
    tree_.reset();
    rclcpp::shutdown();

    delete config_;
    config_ = nullptr;
    node_.reset();
    server_.reset();
    factory_.reset();
    server_thread_.join();
  }

  std::shared_ptr<GenerateResponseActionServer> server_;

protected:
  rclcpp::Node::SharedPtr node_;
  BT::NodeConfiguration *config_;
  std::shared_ptr<BT::BehaviorTreeFactory> factory_;
  std::shared_ptr<BT::Tree> tree_;
  std::thread server_thread_;
};

TEST_F(GenerateChatActionTestFixture, test_chat_ports) {

  std::string xml_txt;

#if defined(BTV3)
  xml_txt =
      R"(
      <root main_tree_to_execute = "MainTree" >
        <BehaviorTree ID="MainTree">
            <GenerateChatCompletions/>
        </BehaviorTree>
      </root>)";
#else
  xml_txt =
      R"(
      <root BTCPP_format="4">
        <BehaviorTree ID="MainTree">
            <GenerateChatCompletions/>
        </BehaviorTree>
      </root>)";
#endif

  tree_ = std::make_shared<BT::Tree>(
      factory_->createTreeFromText(xml_txt, config_->blackboard));
  EXPECT_FALSE(
      tree_->rootNode()
          ->getInput<std::vector<llama_msgs::msg::ChatMessage>>("messages")
          .has_value());

#if defined(BTV3)
  xml_txt =
      R"(
      <root main_tree_to_execute = "MainTree" >
        <BehaviorTree ID="MainTree">
            <GenerateChatCompletions/>
        </BehaviorTree>
      </root>)";
#else
  xml_txt =
      R"(
      <root BTCPP_format="4">
        <BehaviorTree ID="MainTree">
            <GenerateChatCompletions/>
        </BehaviorTree>
      </root>)";
#endif

  tree_ = std::make_shared<BT::Tree>(
      factory_->createTreeFromText(xml_txt, config_->blackboard));
  //   EXPECT_TRUE(
  //       tree_->rootNode()->getInput<std::string>("messages").value().empty());

  //   EXPECT_TRUE(tree_->rootNode()
  //                   ->getInput<std::vector<std::string>>("tools")
  //                   .value()
  //                   .empty());

  EXPECT_EQ(tree_->rootNode()->getInput<std::string>("tool_choice").value(),
            "auto");
}

TEST_F(GenerateChatActionTestFixture, test_chat_tick) {

  std::string xml_txt;

#if defined(BTV3)
  xml_txt =
      R"(
      <root>
        <BehaviorTree ID="MainTree">
            <GenerateChatCompletions choice_message="{response}"/>
        </BehaviorTree>
      </root>)";
#else
  xml_txt =
      R"(
      <root BTCPP_format="4">
        <BehaviorTree ID="MainTree">
            <GenerateChatCompletions choice_message="{response}"/>
        </BehaviorTree>
      </root>)";
#endif

  tree_ = std::make_shared<BT::Tree>(
      factory_->createTreeFromText(xml_txt, config_->blackboard));
  //   EXPECT_TRUE(tree_->rootNode()
  //                   ->getInput<std::vector<llama_msgs::msg::ChatTool>>("tools")
  //                   .value()
  //                   .empty());
  //   EXPECT_TRUE(
  //       tree_->rootNode()
  //           ->getInput<std::vector<llama_msgs::msg::ChatMessage>>("messages")
  //           .value()
  //           .empty());

  rclcpp::Rate rate(30);
  auto start_time = node_->now();
  auto elapsed_time = node_->now() - start_time;
  bool finish = false;
  while (!finish && rclcpp::ok() && elapsed_time.seconds() < 5.0) {
    rclcpp::spin_some(node_->get_node_base_interface());

    finish = tree_->rootNode()->executeTick() != BT::NodeStatus::RUNNING;
    rate.sleep();
    elapsed_time = node_->now() - start_time;
  }

  EXPECT_EQ(tree_->rootNode()->status(), BT::NodeStatus::SUCCESS);

  // Check if the output is correct
  auto response =
      config_->blackboard->get<llama_msgs::msg::ChatMessage>("response");
  EXPECT_EQ(response.content, "This is a test response");
  EXPECT_EQ(response.role, "assistant");
}
