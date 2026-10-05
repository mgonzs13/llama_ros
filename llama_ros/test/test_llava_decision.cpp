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
#include <thread>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "huggingface_hub.h"
#include "llama_utils/llama_params.hpp"
#include "llava_ros/llava.hpp"

class LlavaDecisionTest : public ::testing::Test {
protected:
  void SetUp() override {
    params = std::make_unique<llama_utils::LlamaParams>();

    params->params.n_ctx = 1024;
    params->params.n_batch = 512;
    params->params.n_ubatch = 512;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;

    auto model = huggingface_hub::hf_hub_download_with_shards(
        "ggml-org/tinyopenjev-for-testing-gguf",
        "tinyopenjev-for-testing-Q8_0.gguf");
    ASSERT_TRUE(model.success) << "Failed to download model";
    ASSERT_FALSE(model.path.empty()) << "Model path is empty";

    auto mmproj = huggingface_hub::hf_hub_download_with_shards(
        "ggml-org/tinyopenjev-for-testing-gguf",
        "mmproj-tinyopenjev-for-testing-Q8_0.gguf");
    ASSERT_TRUE(mmproj.success) << "Failed to download mmproj";
    ASSERT_FALSE(mmproj.path.empty()) << "Mmproj path is empty";

    params->params.model.path = model.path;
    params->params.mmproj.path = mmproj.path;

    llava = std::make_unique<llava_ros::Llava>(params->params,
                                               params->system_prompt);

    run_loop_thread = std::thread([this]() {
      try {
        this->llava->run_loop();
      } catch (...) {
      }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void TearDown() override {
    if (llava) {
      llava->cancel();
    }
    if (run_loop_thread.joinable()) {
      run_loop_thread.join();
    }
    llava.reset();
    params.reset();
  }

  std::unique_ptr<llava_ros::Llava> llava;
  std::unique_ptr<llama_utils::LlamaParams> params;
  std::thread run_loop_thread;
};

TEST_F(LlavaDecisionTest, AnswersQuestionWithImage) {
  cv::Mat image(64, 64, CV_8UC3, cv::Scalar(255, 0, 0));
  std::vector<uchar> buffer;
  ASSERT_TRUE(cv::imencode(".jpg", image, buffer));

  ASSERT_TRUE(llava->load_mtmds(
      {std::vector<uint8_t>(buffer.begin(), buffer.end())}, false));

  llama_ros::DecisionQuestion question;
  question.type = llama_ros::DECISION_QUESTION_CHOICE;
  question.instructions = "What is the dominant color?";
  question.keys = {"red", "green", "blue"};

  auto results = llava->evaluate_decisions("An image.", {question}, 1);
  ASSERT_EQ(results.size(), 1u);
  ASSERT_TRUE(results[0].is_ok()) << results[0].error();

  const auto answer = results[0].value();
  EXPECT_EQ(answer.type, llama_ros::DECISION_QUESTION_CHOICE);
  ASSERT_EQ(answer.keys.size(), 3u);
  ASSERT_EQ(answer.probabilities.size(), 3u);

  cv::Mat image2(64, 64, CV_8UC3, cv::Scalar(0, 255, 0));
  std::vector<uchar> buffer2;
  ASSERT_TRUE(cv::imencode(".jpg", image2, buffer2));

  llava->clear_mtmds();
  ASSERT_TRUE(llava->load_mtmds(
      {std::vector<uint8_t>(buffer2.begin(), buffer2.end())}, false));

  auto results2 = llava->evaluate_decisions("Another image.", {question}, 1);
  ASSERT_EQ(results2.size(), 1u);
  ASSERT_TRUE(results2[0].is_ok()) << results2[0].error();
  EXPECT_EQ(results2[0].value().keys.size(), 3u);
}

class LlavaClefDecisionTest : public ::testing::Test {
protected:
  void SetUp() override {
    const char *model_path = std::getenv("LLAMA_ROS_TEST_CLEF_MODEL");
    const char *mmproj_path = std::getenv("LLAMA_ROS_TEST_CLEF_MMPROJ");
    if (model_path == nullptr || model_path[0] == '\0' ||
        mmproj_path == nullptr || mmproj_path[0] == '\0') {
      GTEST_SKIP() << "Set LLAMA_ROS_TEST_CLEF_MODEL and "
                      "LLAMA_ROS_TEST_CLEF_MMPROJ to run";
    }

    params = std::make_unique<llama_utils::LlamaParams>();
    params->params.n_ctx = 8192;
    params->params.n_batch = 2048;
    params->params.n_ubatch = 2048;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.sampling.seed = LLAMA_DEFAULT_SEED;
    params->params.model.path = model_path;
    params->params.mmproj.path = mmproj_path;

    llava = std::make_unique<llava_ros::Llava>(params->params,
                                               params->system_prompt);

    run_loop_thread = std::thread([this]() {
      try {
        this->llava->run_loop();
      } catch (...) {
      }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void TearDown() override {
    if (llava) {
      llava->cancel();
    }
    if (run_loop_thread.joinable()) {
      run_loop_thread.join();
    }
    llava.reset();
    params.reset();
  }

  std::unique_ptr<llava_ros::Llava> llava;
  std::unique_ptr<llama_utils::LlamaParams> params;
  std::thread run_loop_thread;
};

TEST_F(LlavaClefDecisionTest, AnswersQuestionsWithImage) {
  // synthetic scene: a green gradient with a blue rectangle
  cv::Mat image(256, 256, CV_8UC3);
  for (int y = 0; y < image.rows; ++y) {
    image.row(y).setTo(cv::Scalar(0, y, 0));
  }
  cv::rectangle(image, cv::Point(64, 64), cv::Point(192, 192),
                cv::Scalar(255, 0, 0), cv::FILLED);

  std::vector<uchar> buffer;
  ASSERT_TRUE(cv::imencode(".jpg", image, buffer));
  ASSERT_TRUE(llava->load_mtmds(
      {std::vector<uint8_t>(buffer.begin(), buffer.end())}, false));

  llama_ros::DecisionQuestion choice;
  choice.type = llama_ros::DECISION_QUESTION_CHOICE;
  choice.instructions = "What is the dominant color?";
  choice.keys = {"red", "green", "blue"};

  llama_ros::DecisionQuestion score;
  score.type = llama_ros::DECISION_QUESTION_SCORE;
  score.instructions = "How cluttered is the scene?";
  score.descriptions = {"empty", "moderate", "busy"};

  llama_ros::DecisionQuestion noul;
  noul.type = llama_ros::DECISION_QUESTION_NOUL;
  noul.instructions = "Is there a person in the image?";

  auto results =
      llava->evaluate_decisions("A camera frame.", {choice, score, noul}, 1);
  ASSERT_EQ(results.size(), 3u);
  ASSERT_TRUE(results[0].is_ok()) << results[0].error();
  ASSERT_TRUE(results[1].is_ok()) << results[1].error();
  ASSERT_TRUE(results[2].is_ok()) << results[2].error();

  const auto &choice_answer = results[0].value();
  EXPECT_EQ(choice_answer.type, llama_ros::DECISION_QUESTION_CHOICE);
  EXPECT_FALSE(choice_answer.choice.empty());
  ASSERT_EQ(choice_answer.keys.size(), 3u);
  ASSERT_EQ(choice_answer.probabilities.size(), 3u);
  EXPECT_NE(std::find(choice_answer.keys.begin(), choice_answer.keys.end(),
                      choice_answer.choice),
            choice_answer.keys.end());
  float choice_sum = 0.0f;
  for (const float probability : choice_answer.probabilities) {
    EXPECT_GE(probability, 0.0f);
    choice_sum += probability;
  }
  EXPECT_NEAR(choice_sum, 1.0f, 1e-3f);
  EXPECT_GE(choice_answer.confidence, 0.0f);
  EXPECT_LE(choice_answer.confidence, 1.0f);

  const auto &score_answer = results[1].value();
  EXPECT_EQ(score_answer.type, llama_ros::DECISION_QUESTION_SCORE);
  ASSERT_EQ(score_answer.probabilities.size(), 3u);
  EXPECT_GE(score_answer.score, 0.0f);
  EXPECT_LE(score_answer.score, 2.0f);
  EXPECT_GE(score_answer.confidence, 0.0f);
  EXPECT_LE(score_answer.confidence, 1.0f);

  const auto &noul_answer = results[2].value();
  EXPECT_EQ(noul_answer.type, llama_ros::DECISION_QUESTION_NOUL);
  ASSERT_EQ(noul_answer.keys.size(), 2u);
  EXPECT_GE(noul_answer.noul, 0.0f);
  EXPECT_LE(noul_answer.noul, 1.0f);

  // a second request with two images exercises the chunk accumulation path
  llava->clear_mtmds();
  cv::Mat image2(128, 128, CV_8UC3, cv::Scalar(50, 200, 50));
  cv::Mat image3(64, 64, CV_8UC3, cv::Scalar(200, 50, 50));
  std::vector<uchar> buffer2;
  std::vector<uchar> buffer3;
  ASSERT_TRUE(cv::imencode(".jpg", image2, buffer2));
  ASSERT_TRUE(cv::imencode(".jpg", image3, buffer3));
  ASSERT_TRUE(
      llava->load_mtmds({std::vector<uint8_t>(buffer2.begin(), buffer2.end()),
                         std::vector<uint8_t>(buffer3.begin(), buffer3.end())},
                        false));

  auto multi = llava->evaluate_decisions("Two camera frames.", {choice}, 2);
  ASSERT_EQ(multi.size(), 1u);
  ASSERT_TRUE(multi[0].is_ok()) << multi[0].error();
  EXPECT_FALSE(multi[0].value().choice.empty());

  // a solid color must be grounded by a single image
  auto evaluate_solid = [&](const cv::Scalar &color) {
    cv::Mat solid(256, 256, CV_8UC3, color);
    std::vector<uchar> buffer_solid;
    EXPECT_TRUE(cv::imencode(".jpg", solid, buffer_solid));
    llava->clear_mtmds();
    EXPECT_TRUE(llava->load_mtmds(
        {std::vector<uint8_t>(buffer_solid.begin(), buffer_solid.end())},
        false));
    return llava->evaluate_decisions("A camera frame.", {choice}, 1);
  };

  auto red = evaluate_solid(cv::Scalar(0, 0, 255));
  ASSERT_EQ(red.size(), 1u);
  ASSERT_TRUE(red[0].is_ok()) << red[0].error();
  EXPECT_EQ(red[0].value().choice, "red");

  auto blue = evaluate_solid(cv::Scalar(255, 0, 0));
  ASSERT_EQ(blue.size(), 1u);
  ASSERT_TRUE(blue[0].is_ok()) << blue[0].error();
  EXPECT_EQ(blue[0].value().choice, "blue");

  // two images: the decision must not depend on their order
  auto evaluate_pair = [&](const cv::Scalar &first, const cv::Scalar &second) {
    cv::Mat a(256, 256, CV_8UC3, first);
    cv::Mat b(256, 256, CV_8UC3, second);
    std::vector<uchar> buffer_a;
    std::vector<uchar> buffer_b;
    EXPECT_TRUE(cv::imencode(".jpg", a, buffer_a));
    EXPECT_TRUE(cv::imencode(".jpg", b, buffer_b));
    llava->clear_mtmds();
    EXPECT_TRUE(llava->load_mtmds(
        {std::vector<uint8_t>(buffer_a.begin(), buffer_a.end()),
         std::vector<uint8_t>(buffer_b.begin(), buffer_b.end())},
        false));
    return llava->evaluate_decisions("Two camera frames.", {choice}, 2);
  };

  auto red_blue = evaluate_pair(cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0));
  auto blue_red = evaluate_pair(cv::Scalar(255, 0, 0), cv::Scalar(0, 0, 255));
  ASSERT_EQ(red_blue.size(), 1u);
  ASSERT_EQ(blue_red.size(), 1u);
  ASSERT_TRUE(red_blue[0].is_ok()) << red_blue[0].error();
  ASSERT_TRUE(blue_red[0].is_ok()) << blue_red[0].error();
  // both images are processed; the first one grounds the answer (the old
  // last-image-only bug would swap these)
  EXPECT_EQ(red_blue[0].value().choice, "red");
  EXPECT_EQ(blue_red[0].value().choice, "blue");
}
