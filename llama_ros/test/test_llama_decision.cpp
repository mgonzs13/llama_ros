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
#include <cstdlib>
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
  question.keys = {"red", "green"};
  question.descriptions = {"The color of the light.", "The color of grass."};

  auto results =
      llama->evaluate_decisions("The traffic light is red.", {question});
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].is_ok()) << results[0].error();

  const auto answer = results[0].value();
  EXPECT_EQ(answer.type, llama_ros::DECISION_QUESTION_CHOICE);
  ASSERT_EQ(answer.keys.size(), 2u);
  ASSERT_EQ(answer.probabilities.size(), 2u);

  float sum = 0.0f;
  for (const float probability : answer.probabilities) {
    EXPECT_GE(probability, 0.0f);
    sum += probability;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-4);

  const auto best = std::max_element(answer.probabilities.begin(),
                                     answer.probabilities.end());
  const size_t best_index =
      static_cast<size_t>(best - answer.probabilities.begin());
  EXPECT_EQ(answer.choice, answer.keys[best_index]);

  EXPECT_GE(answer.confidence, 0.0f);
  EXPECT_LE(answer.confidence, 1.0f);
}

TEST_F(LlamaDecisionTest, AnswersScoreQuestion) {
  llama_ros::DecisionQuestion question;
  question.type = llama_ros::DECISION_QUESTION_SCORE;
  question.instructions = "How urgent is the situation?";
  question.descriptions = {"none", "low", "high", "critical"};

  auto results = llama->evaluate_decisions(
      "The traffic light is red and the car is moving.", {question});
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].is_ok()) << results[0].error();

  const auto answer = results[0].value();
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

  auto results =
      llama->evaluate_decisions("The traffic light is red.", {question});
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].is_ok()) << results[0].error();

  const auto answer = results[0].value();
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

  auto choice_results = llama->evaluate_decisions("Some state.", {choice});
  ASSERT_EQ(choice_results.size(), 1u);
  EXPECT_TRUE(choice_results[0].is_error());

  llama_ros::DecisionQuestion score;
  score.type = llama_ros::DECISION_QUESTION_SCORE;
  score.instructions = "Rate this.";
  score.descriptions = {"only one level"};

  auto score_results = llama->evaluate_decisions("Some state.", {score});
  ASSERT_EQ(score_results.size(), 1u);
  EXPECT_TRUE(score_results[0].is_error());
}

TEST_F(LlamaDecisionTest, RejectsImagesForNonImageModel) {
  llama_ros::DecisionQuestion question;
  question.type = llama_ros::DECISION_QUESTION_CHOICE;
  question.instructions = "Pick one.";
  question.keys = {"a", "b"};

  auto results = llama->evaluate_decisions("Some state.", {question}, 1);
  ASSERT_EQ(results.size(), 1u);
  EXPECT_TRUE(results[0].is_error());
}

TEST_F(LlamaDecisionTest, AnswersMultipleQuestions) {
  llama_ros::DecisionQuestion choice;
  choice.type = llama_ros::DECISION_QUESTION_CHOICE;
  choice.instructions = "Pick the best color.";
  choice.keys = {"red", "green"};

  llama_ros::DecisionQuestion noul;
  noul.type = llama_ros::DECISION_QUESTION_NOUL;
  noul.instructions = "The light is red.";

  auto results =
      llama->evaluate_decisions("The traffic light is red.", {choice, noul});
  ASSERT_EQ(results.size(), 2u);
  EXPECT_TRUE(results[0].is_ok()) << results[0].error();
  EXPECT_TRUE(results[1].is_ok()) << results[1].error();
  EXPECT_EQ(results[0].value().type, llama_ros::DECISION_QUESTION_CHOICE);
  EXPECT_EQ(results[1].value().type, llama_ros::DECISION_QUESTION_NOUL);
}

TEST_F(LlamaDecisionTest, MetadataHandlesArrayValues) {
  // modern-bert stores feed_forward_length as a per-layer array; the old
  // std::stoi-based parsing threw on it
  llama_ros::Metadata metadata;
  EXPECT_NO_THROW(metadata = llama->get_metadata());

  EXPECT_TRUE(metadata.decision.enabled);
  EXPECT_EQ(metadata.decision.type, "laya");
  EXPECT_EQ(metadata.decision.max_head_tokens, 192u);

  const char *systemone =
      llama_model_chat_template(llama->get_model(), "systemone");
  ASSERT_NE(systemone, nullptr);
  EXPECT_EQ(metadata.decision.systemone_template, std::string(systemone));
}

/**
 * @brief Test suite for the label-logit decision path (OpenJEV).
 */
class LlamaOpenJevTest : public ::testing::Test {
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
        "ggml-org/tinyopenjev-for-testing-gguf",
        "tinyopenjev-for-testing-Q8_0.gguf");

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

TEST_F(LlamaOpenJevTest, IsNotEmbeddingMode) {
  EXPECT_TRUE(llama->is_decision());
  EXPECT_FALSE(llama->is_embedding());
}

TEST_F(LlamaOpenJevTest, AnswersChoiceQuestion) {
  llama_ros::DecisionQuestion question;
  question.type = llama_ros::DECISION_QUESTION_CHOICE;
  question.instructions = "Pick the best color.";
  question.keys = {"red", "green"};

  auto results =
      llama->evaluate_decisions("The traffic light is red.", {question});
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].is_ok()) << results[0].error();

  const auto answer = results[0].value();
  EXPECT_EQ(answer.type, llama_ros::DECISION_QUESTION_CHOICE);
  EXPECT_FALSE(answer.choice.empty());

  float sum = 0.0f;
  for (const float probability : answer.probabilities) {
    sum += probability;
  }
  EXPECT_NEAR(sum, 1.0f, 1e-3f);

  ASSERT_EQ(answer.keys.size(), 2u);
  const auto best = std::max_element(answer.probabilities.begin(),
                                     answer.probabilities.end());
  const size_t best_index =
      static_cast<size_t>(best - answer.probabilities.begin());
  EXPECT_EQ(answer.choice, answer.keys[best_index]);
}

TEST_F(LlamaOpenJevTest, ReportsDecisionMetadata) {
  llama_ros::Metadata metadata = llama->get_metadata();

  EXPECT_TRUE(metadata.decision.enabled);
  EXPECT_EQ(metadata.decision.type, "openjev");
  ASSERT_FALSE(metadata.decision.temperature_names.empty());
  EXPECT_EQ(metadata.decision.temperature_names.size(),
            metadata.decision.temperatures.size());
  EXPECT_FALSE(metadata.decision.systemone_template.empty());

  EXPECT_EQ(metadata.tokenizer.chat_templates,
            (std::vector<std::string>{"systemone"}));

  // the tiny OpenJEV chat template is over the old 4096 cap
  EXPECT_GT(metadata.tokenizer.chat_template.size(), 4096u);

  const char *systemone =
      llama_model_chat_template(llama->get_model(), "systemone");
  ASSERT_NE(systemone, nullptr);
  EXPECT_EQ(metadata.decision.systemone_template, std::string(systemone));
  EXPECT_EQ(metadata.decision.systemone_template.find('\0'), std::string::npos);
  for (const auto &name : metadata.decision.temperature_names) {
    EXPECT_EQ(name.find('\0'), std::string::npos);
  }
  for (const auto &name : metadata.tokenizer.chat_templates) {
    EXPECT_EQ(name.find('\0'), std::string::npos);
  }

  EXPECT_EQ(metadata.sampling.top_k, 20);
  EXPECT_NEAR(metadata.sampling.top_p, 0.95f, 1e-4f);
  EXPECT_NEAR(metadata.sampling.temp, 1.0f, 1e-4f);
}

/**
 * @brief Test suite for decision types without tiny GGUF models.
 *
 * Set LLAMA_ROS_TEST_MODEL to the path of a decision model to run it.
 */
class LlamaEnvDecisionTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *env_var = std::getenv("LLAMA_ROS_TEST_MODEL");
    if (env_var == nullptr || env_var[0] == '\0') {
      GTEST_SKIP() << "Set LLAMA_ROS_TEST_MODEL to a decision GGUF to run";
    }

    params = std::make_unique<llama_utils::LlamaParams>();
    params->params.n_ctx = 2048;
    params->params.n_batch = 1024;
    params->params.n_ubatch = 1024;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.model.path = env_var;

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

TEST_F(LlamaEnvDecisionTest, AnswersAllQuestionTypes) {
  llama_ros::DecisionQuestion choice;
  choice.type = llama_ros::DECISION_QUESTION_CHOICE;
  choice.instructions = "Pick the best option.";
  choice.keys = {"a", "b"};

  llama_ros::DecisionQuestion score;
  score.type = llama_ros::DECISION_QUESTION_SCORE;
  score.instructions = "How good is it?";
  score.descriptions = {"bad", "ok", "good"};

  llama_ros::DecisionQuestion noul;
  noul.type = llama_ros::DECISION_QUESTION_NOUL;
  noul.instructions = "It is good.";

  auto results =
      llama->evaluate_decisions("A test state.", {choice, score, noul});
  ASSERT_EQ(results.size(), 3u);
  EXPECT_TRUE(results[0].is_ok()) << results[0].error();
  EXPECT_TRUE(results[1].is_ok()) << results[1].error();
  EXPECT_TRUE(results[2].is_ok()) << results[2].error();
}

TEST_F(LlamaDecisionTest, ClefJointFillAssignsOrders) {
  llama_ros::DecisionModelConfig config;
  config.type = COMMON_DECISION_TYPE_CLEF;
  config.n_options_max = 255;
  config.noul_true_first = true;
  config.choice_sorted = true;
  llama_ros::DecisionModel model(config, nullptr);

  llama_ros::DecisionQuestion question;
  question.id = "0";
  question.type = llama_ros::DECISION_QUESTION_CHOICE;
  question.instructions = "Pick one.";
  question.keys = {"red", "blue"};

  const std::string prompt =
      "state<<clef:sep>><<clef:question>>Pick one."
      "<<clef:sep>><<clef:option>>red<<clef:sep>><<clef:option>>blue";

  // text-only path: head_end = 0
  std::vector<llama_token> tokens;
  llama_ros::DecisionTaskMeta meta;
  model.fill_task_joint(llama->get_vocab(), {question}, prompt, 0, tokens,
                        meta);
  EXPECT_EQ(meta.n_scores, 2);
  ASSERT_EQ(meta.order.size(), tokens.size());
  EXPECT_NE(std::find(meta.order.begin(), meta.order.end(), 2),
            meta.order.end());
  EXPECT_EQ(std::count(meta.order.begin(), meta.order.end(), 4), 2);

  // mixed path: the head entries stay at order 0 and are preserved
  std::vector<llama_token> mixed_tokens = {11, 12, 13};
  llama_ros::DecisionTaskMeta mixed_meta;
  model.fill_task_joint(llama->get_vocab(), {question}, prompt, 1, mixed_tokens,
                        mixed_meta);
  ASSERT_GE(mixed_tokens.size(), 3u);
  EXPECT_EQ(mixed_tokens[0], 11);
  EXPECT_EQ(mixed_tokens[1], 12);
  EXPECT_EQ(mixed_tokens[2], 13);
  ASSERT_EQ(mixed_meta.order.size(), mixed_tokens.size());
  EXPECT_EQ(mixed_meta.order[0], 0);
  EXPECT_EQ(mixed_meta.order[1], 0);
  EXPECT_EQ(mixed_meta.order[2], 0);
  EXPECT_EQ(mixed_meta.n_scores, 2);
}
