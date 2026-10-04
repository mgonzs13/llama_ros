// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
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

#include <algorithm>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>

#include "huggingface_hub.h"
#include "llama_ros/llama.hpp"
#include "llama_utils/llama_params.hpp"

/**
 * @brief Test suite for decision (Laya) functionality.
 *
 * This test suite verifies the decision evaluation capabilities of the
 * LLM, which requires a Laya decision model.
 */
class LlamaDecisionTest : public ::testing::Test {
protected:
  void SetUp() override {
    params = std::make_unique<llama_utils::LlamaParams>();

    params->params.n_ctx = 1024;
    params->params.n_batch = 512;
    params->params.n_ubatch = 512;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.sampling.seed = LLAMA_DEFAULT_SEED;

    auto result = huggingface_hub::hf_hub_download_with_shards(
        "ggml-org/tinylaya-for-testing-gguf", "tinylaya-for-testing-Q8_0.gguf");

    ASSERT_TRUE(result.success) << "Failed to download model";
    ASSERT_FALSE(result.path.empty()) << "Model path is empty";

    params->params.model.path = result.path;

    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
    ASSERT_NE(llama, nullptr);

    run_loop_thread = std::thread([this]() {
      try {
        this->llama->run_loop();
      } catch (const std::exception &e) {
        // Log error but don't fail the test here
      } catch (...) {
        // Handle unknown exceptions
      }
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void TearDown() override {
    if (llama) {
      llama->cancel();
    }

    if (run_loop_thread.joinable()) {
      try {
        run_loop_thread.join();
      } catch (const std::exception &e) {
        // Ignore join errors during cleanup
      }
    }

    llama.reset();
    params.reset();
  }

  std::unique_ptr<llama_ros::Llama> llama;
  std::unique_ptr<llama_utils::LlamaParams> params;
  std::thread run_loop_thread;
};

TEST_F(LlamaDecisionTest, DetectsDecisionModel) {
  EXPECT_TRUE(llama->is_decision());
  EXPECT_TRUE(llama->is_embedding());
}

TEST_F(LlamaDecisionTest, AnswersChoiceQuestion) {
  llama_ros::DecisionQuestion question;
  question.type = llama_ros::DECISION_QUESTION_CHOICE;
  question.instructions = "Pick the best color.";
  question.state = "The traffic light is red.";
  question.keys = {"red", "green"};
  question.descriptions = {"The color of the light.", "The color of grass."};

  auto result = llama->evaluate_decision(question);
  ASSERT_TRUE(result.is_ok()) << result.error();

  const auto answer = result.value();
  EXPECT_EQ(answer.type, llama_ros::DECISION_QUESTION_CHOICE);
  ASSERT_EQ(answer.keys.size(), 2u);
  ASSERT_EQ(answer.probabilities.size(), 2u);

  float sum = 0.0f;
  for (const float probability : answer.probabilities) {
    EXPECT_GE(probability, 0.0f);
    sum += probability;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-4);

  const size_t best = std::max_element(answer.probabilities.begin(),
                                       answer.probabilities.end()) -
                      answer.probabilities.begin();
  EXPECT_EQ(answer.choice, answer.keys[best]);
  EXPECT_GE(answer.confidence, 0.0f);
  EXPECT_LE(answer.confidence, 1.0f);
}

TEST_F(LlamaDecisionTest, AnswersScoreQuestion) {
  llama_ros::DecisionQuestion question;
  question.type = llama_ros::DECISION_QUESTION_SCORE;
  question.instructions = "How urgent is the situation?";
  question.state = "The traffic light is red and the car is moving.";
  question.descriptions = {"none", "low", "high", "critical"};

  auto result = llama->evaluate_decision(question);
  ASSERT_TRUE(result.is_ok()) << result.error();

  const auto answer = result.value();
  EXPECT_EQ(answer.type, llama_ros::DECISION_QUESTION_SCORE);
  ASSERT_EQ(answer.probabilities.size(), 4u);

  float sum = 0.0f;
  for (const float probability : answer.probabilities) {
    EXPECT_GE(probability, 0.0f);
    sum += probability;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-4);

  EXPECT_GE(answer.score, 0.0f);
  EXPECT_LE(answer.score, 3.0f);
  EXPECT_GE(answer.confidence, 0.0f);
  EXPECT_LE(answer.confidence, 1.0f);
}

TEST_F(LlamaDecisionTest, AnswersNoulQuestion) {
  llama_ros::DecisionQuestion question;
  question.type = llama_ros::DECISION_QUESTION_NOUL;
  question.instructions = "The light is green.";
  question.state = "The traffic light is red.";

  auto result = llama->evaluate_decision(question);
  ASSERT_TRUE(result.is_ok()) << result.error();

  const auto answer = result.value();
  EXPECT_EQ(answer.type, llama_ros::DECISION_QUESTION_NOUL);
  ASSERT_EQ(answer.keys.size(), 2u);
  ASSERT_EQ(answer.probabilities.size(), 2u);
  EXPECT_EQ(answer.keys[0], "false");
  EXPECT_EQ(answer.keys[1], "true");

  float sum = 0.0f;
  for (const float probability : answer.probabilities) {
    EXPECT_GE(probability, 0.0f);
    sum += probability;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-4);
  EXPECT_GE(answer.noul, 0.0f);
  EXPECT_LE(answer.noul, 1.0f);
}

TEST_F(LlamaDecisionTest, RejectsInvalidQuestions) {
  llama_ros::DecisionQuestion choice;
  choice.type = llama_ros::DECISION_QUESTION_CHOICE;
  choice.instructions = "Pick one.";
  choice.state = "Some state.";
  EXPECT_TRUE(llama->evaluate_decision(choice).is_error());

  llama_ros::DecisionQuestion score;
  score.type = llama_ros::DECISION_QUESTION_SCORE;
  score.instructions = "Rate this.";
  score.state = "Some state.";
  score.descriptions = {"only one level"};
  EXPECT_TRUE(llama->evaluate_decision(score).is_error());
}
