
// MIT License
//
// Copyright (c) 2025 Alberto J. Tudela Roldán
// Copyright (c) 2025 Grupo Avispa, DTE, Universidad de Málaga
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

#include <gtest/gtest.h>
#include <memory>
#include <string>

#if defined(BTV3)
#include "behaviortree_cpp_v3/bt_factory.h"
#else
#include "behaviortree_cpp/bt_factory.h"
#endif

#include "rclcpp/executors.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "llama_bt/action/bt_types.hpp"
#include "llama_bt/action/generate_response_action.hpp"
#include "utils/test_action_server.hpp"

class GenerateResponseActionServer
    : public TestActionServer<llama_msgs::action::GenerateResponse> {
public:
  GenerateResponseActionServer() : TestActionServer("generate_response") {}

protected:
  void
  execute(const typename std::shared_ptr<
          rclcpp_action::ServerGoalHandle<llama_msgs::action::GenerateResponse>>
              goal_handle) override {
    llama_msgs::action::GenerateResponse::Result::SharedPtr result =
        std::make_shared<llama_msgs::action::GenerateResponse::Result>();
    result->response.text = "This is a test response";
    bool return_success = getReturnSuccess();
    if (return_success) {
      goal_handle->succeed(result);
    } else {
      goal_handle->abort(result);
    }
  }
};

class GenerateResponseActionTestFixture : public ::testing::Test {
public:
  void SetUp() override {
    rclcpp::init(0, nullptr);

    node_ = std::make_shared<rclcpp::Node>("generate_response_test_fixture");
    executor_ = std::make_unique<rclcpp::executors::SingleThreadedExecutor>();
    executor_->add_node(node_);
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
      return std::make_unique<llama_bt::GenerateResponseAction>(
          name, "generate_response", config);
    };

    factory_->registerBuilder<llama_bt::GenerateResponseAction>(
        "GenerateResponse", builder);

    server_ = std::make_shared<GenerateResponseActionServer>();
    server_thread_ = std::thread([this]() { rclcpp::spin(server_); });
  }

  void TearDown() override {
    tree_.reset();
    executor_.reset();
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
  std::unique_ptr<rclcpp::executors::SingleThreadedExecutor> executor_;
  BT::NodeConfiguration *config_;
  std::shared_ptr<BT::BehaviorTreeFactory> factory_;
  std::shared_ptr<BT::Tree> tree_;
  std::thread server_thread_;
};

TEST_F(GenerateResponseActionTestFixture, test_ports) {

  std::string xml_txt;

#if defined(BTV3)
  xml_txt =
      R"(
      <root main_tree_to_execute = "MainTree" >
        <BehaviorTree ID="MainTree">
            <GenerateResponse/>
        </BehaviorTree>
      </root>)";
#else
  xml_txt =
      R"(
      <root BTCPP_format="4">
        <BehaviorTree ID="MainTree">
            <GenerateResponse/>
        </BehaviorTree>
      </root>)";
#endif

  tree_ = std::make_shared<BT::Tree>(
      factory_->createTreeFromText(xml_txt, config_->blackboard));
  EXPECT_FALSE(tree_->rootNode()->getInput<bool>("reset").value());

#if defined(BTV3)
  xml_txt =
      R"(
      <root main_tree_to_execute = "MainTree" >
        <BehaviorTree ID="MainTree">
            <GenerateResponse prompt="" stop=""/>
        </BehaviorTree>
      </root>)";
#else
  xml_txt =
      R"(
      <root BTCPP_format="4">
        <BehaviorTree ID="MainTree">
            <GenerateResponse prompt="" stop=""/>
        </BehaviorTree>
      </root>)";
#endif

  tree_ = std::make_shared<BT::Tree>(
      factory_->createTreeFromText(xml_txt, config_->blackboard));
  EXPECT_TRUE(
      tree_->rootNode()->getInput<std::string>("prompt").value().empty());

  EXPECT_TRUE(tree_->rootNode()
                  ->getInput<std::vector<std::string>>("stop")
                  .value()
                  .empty());

  EXPECT_FALSE(tree_->rootNode()->getInput<bool>("reset").value());

#if defined(BTV3)
  xml_txt =
      R"(
      <root main_tree_to_execute = "MainTree" >
        <BehaviorTree ID="MainTree">
            <GenerateResponse prompt="This is a test" stop="This;test" reset="true" response="{response}"/>
        </BehaviorTree>
      </root>)";
#else
  xml_txt =
      R"(
      <root BTCPP_format="4">
        <BehaviorTree ID="MainTree">
            <GenerateResponse prompt="This is a test" stop="This;test" reset="true" response="{response}"/>
        </BehaviorTree>
      </root>)";
#endif

  tree_ = std::make_shared<BT::Tree>(
      factory_->createTreeFromText(xml_txt, config_->blackboard));
  EXPECT_EQ(tree_->rootNode()->getInput<std::string>("prompt"),
            "This is a test");

  auto stop_optional =
      tree_->rootNode()->getInput<std::vector<std::string>>("stop");
  ASSERT_TRUE(stop_optional.has_value());
  std::vector<std::string> stop = stop_optional.value();
  EXPECT_EQ(stop.size(), 2);
  EXPECT_EQ(stop[0], "This");
  EXPECT_EQ(stop[1], "test");
  EXPECT_TRUE(tree_->rootNode()->getInput<bool>("reset").value());
}

TEST_F(GenerateResponseActionTestFixture, test_tick) {

  std::string xml_txt;

#if defined(BTV3)
  xml_txt =
      R"(
      <root>
        <BehaviorTree ID="MainTree">
            <GenerateResponse prompt="" stop="" response="{response}"/>
        </BehaviorTree>
      </root>)";
#else
  xml_txt =
      R"(
      <root BTCPP_format="4">
        <BehaviorTree ID="MainTree">
            <GenerateResponse prompt="" stop="" response="{response}"/>
        </BehaviorTree>
      </root>)";
#endif

  tree_ = std::make_shared<BT::Tree>(
      factory_->createTreeFromText(xml_txt, config_->blackboard));
  EXPECT_TRUE(
      tree_->rootNode()->getInput<std::string>("prompt").value().empty());
  EXPECT_TRUE(tree_->rootNode()
                  ->getInput<std::vector<std::string>>("stop")
                  .value()
                  .empty());
  EXPECT_FALSE(tree_->rootNode()->getInput<bool>("reset").value());

  rclcpp::Rate rate(30);
  auto start_time = node_->now();
  auto elapsed_time = node_->now() - start_time;
  bool finish = false;
  while (!finish && rclcpp::ok() && elapsed_time.seconds() < 5.0) {
    executor_->spin_some();

    finish = tree_->rootNode()->executeTick() != BT::NodeStatus::RUNNING;
    rate.sleep();
    elapsed_time = node_->now() - start_time;
  }

  EXPECT_EQ(tree_->rootNode()->status(), BT::NodeStatus::SUCCESS);

  // Check if the output is correct
  auto response = config_->blackboard->get<std::string>("response");
  EXPECT_EQ(response, "This is a test response");
}
