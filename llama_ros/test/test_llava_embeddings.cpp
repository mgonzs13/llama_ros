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

#include <chrono>
#include <cmath>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include <opencv2/imgcodecs.hpp>

#include "huggingface_hub.h"
#include "llama_utils/llama_params.hpp"
#include "llava_ros/llava.hpp"

namespace {
constexpr const char *kModelRepo = "ggml-org/embeddinggemma-2-GGUF";
constexpr const char *kModelFile = "embeddinggemma-2-Q8_0.gguf";
constexpr const char *kMmprojFile = "mmproj-embeddinggemma-2-Q8_0.gguf";
constexpr size_t kEmbeddingDim = 768;
constexpr const char *kMediaMarker = "<__media__>";
} // namespace

class LlavaEmbeddingsTest : public ::testing::Test {
protected:
  void SetUp() override {
    params = std::make_unique<llama_utils::LlamaParams>();

    params->params.n_ctx = 8192;
    params->params.n_batch = 1024;
    params->params.n_ubatch = 1024;
    params->params.n_parallel = 1;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.sampling.seed = LLAMA_DEFAULT_SEED;
    params->params.embedding = true;
    params->params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    auto model =
        huggingface_hub::hf_hub_download_with_shards(kModelRepo, kModelFile);
    ASSERT_TRUE(model.success) << "Failed to download model";
    ASSERT_FALSE(model.path.empty()) << "Model path is empty";

    auto mmproj =
        huggingface_hub::hf_hub_download_with_shards(kModelRepo, kMmprojFile);
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

  std::vector<uint8_t> make_image(const cv::Scalar &color) {
    cv::Mat image(64, 64, CV_8UC3, color);
    std::vector<uchar> buffer;
    EXPECT_TRUE(cv::imencode(".jpg", image, buffer));
    return std::vector<uint8_t>(buffer.begin(), buffer.end());
  }

  static void
  expect_embedding(const llama_ros::ServerTaskResultEmbedding &emb) {
    ASSERT_EQ(emb.embeddings.size(), 1u);
    const auto &vec = emb.embeddings.front();
    ASSERT_EQ(vec.size(), kEmbeddingDim);
    float norm = 0.0f;
    for (const float value : vec) {
      ASSERT_TRUE(std::isfinite(value));
      norm += value * value;
    }
    EXPECT_NEAR(std::sqrt(norm), 1.0f, 1e-3f);
  }

  std::unique_ptr<llava_ros::Llava> llava;
  std::unique_ptr<llama_utils::LlamaParams> params;
  std::thread run_loop_thread;
};

TEST_F(LlavaEmbeddingsTest, TextOnlyEmbeddingHasFullOutputDimension) {
  auto result = llava->generate_embeddings(
      "task: sentence similarity | query: hello world");
  ASSERT_TRUE(result.is_ok()) << result.error();
  expect_embedding(result.value());
}

TEST_F(LlavaEmbeddingsTest, MarkerCountMismatchFails) {
  ASSERT_TRUE(llava->load_mtmds({make_image(cv::Scalar(0, 0, 255))}, false));
  auto result = llava->generate_embeddings("no marker in this prompt");
  EXPECT_TRUE(result.is_error());
}

TEST_F(LlavaEmbeddingsTest, MarkerWithoutMediaFails) {
  auto result = llava->generate_embeddings(std::string("a prompt with ") +
                                           kMediaMarker + " but no media");
  EXPECT_TRUE(result.is_error());
}

class LlavaEmbeddingsParallelTest : public ::testing::Test {
protected:
  void SetUp() override {
    params = std::make_unique<llama_utils::LlamaParams>();

    params->params.n_ctx = 8192;
    params->params.n_batch = 1024;
    params->params.n_ubatch = 1024;
    params->params.n_parallel = 2;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.sampling.seed = LLAMA_DEFAULT_SEED;
    params->params.embedding = true;
    params->params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    auto model =
        huggingface_hub::hf_hub_download_with_shards(kModelRepo, kModelFile);
    ASSERT_TRUE(model.success) << "Failed to download model";
    ASSERT_FALSE(model.path.empty()) << "Model path is empty";

    auto mmproj =
        huggingface_hub::hf_hub_download_with_shards(kModelRepo, kMmprojFile);
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

TEST_F(LlavaEmbeddingsParallelTest, MediaWithMultipleParallelSlotsFails) {
  cv::Mat image(64, 64, CV_8UC3, cv::Scalar(0, 0, 255));
  std::vector<uchar> buffer;
  ASSERT_TRUE(cv::imencode(".jpg", image, buffer));
  ASSERT_TRUE(llava->load_mtmds(
      {std::vector<uint8_t>(buffer.begin(), buffer.end())}, false));

  auto result = llava->generate_embeddings("a solid color <__media__>");
  EXPECT_TRUE(result.is_error());
}

TEST_F(LlavaEmbeddingsTest, MediaIsDecodedWithWholePrompt) {
  ASSERT_TRUE(llava->load_mtmds({make_image(cv::Scalar(0, 0, 255))}, false));

  auto prefix_first = llava->generate_embeddings(
      "the quick brown fox jumps over the lazy dog <__media__>");
  ASSERT_TRUE(prefix_first.is_ok()) << prefix_first.error();
  expect_embedding(prefix_first.value());

  auto media_first = llava->generate_embeddings(
      "<__media__> the quick brown fox jumps over the lazy dog");
  ASSERT_TRUE(media_first.is_ok()) << media_first.error();
  expect_embedding(media_first.value());

  // the decode window must cover the whole prompt regardless of marker
  // position; a separate pre-media decode would shrink the prefix-first window
  EXPECT_NEAR(prefix_first.value().n_tokens, media_first.value().n_tokens, 4);
}

TEST_F(LlavaEmbeddingsTest, InterleavedEmbeddingIsDeterministic) {
  ASSERT_TRUE(llava->load_mtmds({make_image(cv::Scalar(255, 0, 0))}, false));

  auto first =
      llava->generate_embeddings("a photo <__media__> of a solid color");
  ASSERT_TRUE(first.is_ok()) << first.error();

  auto second =
      llava->generate_embeddings("a photo <__media__> of a solid color");
  ASSERT_TRUE(second.is_ok()) << second.error();

  expect_embedding(first.value());
  expect_embedding(second.value());
  ASSERT_EQ(first.value().embeddings.size(), second.value().embeddings.size());
  ASSERT_EQ(first.value().embeddings.front().size(),
            second.value().embeddings.front().size());
  for (size_t i = 0; i < first.value().embeddings.front().size(); ++i) {
    EXPECT_NEAR(first.value().embeddings.front()[i],
                second.value().embeddings.front()[i], 1e-6f);
  }
}

TEST_F(LlavaEmbeddingsTest, DifferentImagesProduceDifferentEmbeddings) {
  ASSERT_TRUE(llava->load_mtmds({make_image(cv::Scalar(0, 0, 255))}, false));
  auto red = llava->generate_embeddings("a solid color <__media__>");
  ASSERT_TRUE(red.is_ok()) << red.error();

  llava->clear_mtmds();
  ASSERT_TRUE(llava->load_mtmds({make_image(cv::Scalar(255, 0, 0))}, false));
  auto blue = llava->generate_embeddings("a solid color <__media__>");
  ASSERT_TRUE(blue.is_ok()) << blue.error();

  expect_embedding(red.value());
  expect_embedding(blue.value());
  EXPECT_NE(red.value().embeddings, blue.value().embeddings);
}
