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
