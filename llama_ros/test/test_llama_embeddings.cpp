// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
// Copyright (c) 2026 Alejandro González Cantón
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

#include "huggingface_hub.h"
#include "llama_ros/llama.hpp"
#include "llama_utils/llama_params.hpp"

/**
 * @brief Test suite for embeddings functionality.
 *
 * This test suite verifies the embedding generation capabilities
 * of the LLM, which requires the model to be loaded in embedding mode.
 */
class LlamaEmbeddingsTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize test parameters
    params = std::make_unique<llama_utils::LlamaParams>();

    // Set up basic common_params structure for embeddings
    params->params.n_ctx = 512;
    params->params.n_batch = 128;
    params->params.n_ubatch = 128;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.sampling.seed = LLAMA_DEFAULT_SEED;

    // Enable embedding mode
    params->params.embedding = true;
    params->params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    // Download embedding model from HuggingFace
    auto result = huggingface_hub::hf_hub_download_with_shards(
        "bartowski/SmolLM2-135M-Instruct-GGUF",
        "SmolLM2-135M-Instruct-Q6_K.gguf");

    ASSERT_TRUE(result.success) << "Failed to download model";
    ASSERT_FALSE(result.path.empty()) << "Model path is empty";

    // Set the downloaded model path
    params->params.model.path = result.path;

    // Create Llama object in embedding mode
    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
    ASSERT_NE(llama, nullptr);
    ASSERT_TRUE(llama->is_embedding());

    // Start the run_loop thread
    run_loop_thread = std::thread([this]() {
      try {
        this->llama->run_loop();
      } catch (const std::exception &e) {
        // Log error but don't fail the test here
      } catch (...) {
        // Handle unknown exceptions
      }
    });

    // Give the thread a moment to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  void TearDown() override {
    // Cancel any ongoing operations and stop the run_loop
    if (llama) {
      llama->cancel();
    }

    // Wait for the run_loop thread to finish
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

/**
 * @brief Test basic embedding generation.
 */
TEST_F(LlamaEmbeddingsTest, CanGenerateEmbeddings) {
  const std::string text = "Hello world";

  auto result = llama->generate_embeddings(text);

  ASSERT_TRUE(result.is_ok()) << "Failed to generate embeddings";

  auto embedding_result = result.value();

  // Verify embeddings were generated
  EXPECT_FALSE(embedding_result.embeddings.empty());
  EXPECT_GT(embedding_result.n_tokens, 0);

  // Each embedding should be a vector of floats
  for (const auto &emb : embedding_result.embeddings) {
    EXPECT_FALSE(emb.empty()) << "Embedding vector should not be empty";
  }
}

/**
 * @brief Test embeddings for different texts are different.
 */
TEST_F(LlamaEmbeddingsTest, DifferentTextsProduceDifferentEmbeddings) {
  const std::string text1 = "The quick brown fox";
  const std::string text2 = "A completely different sentence";

  auto result1 = llama->generate_embeddings(text1);
  auto result2 = llama->generate_embeddings(text2);

  ASSERT_TRUE(result1.is_ok());
  ASSERT_TRUE(result2.is_ok());

  auto embedding1 = result1.value();
  auto embedding2 = result2.value();

  ASSERT_FALSE(embedding1.embeddings.empty());
  ASSERT_FALSE(embedding2.embeddings.empty());

  // Embeddings should be different
  bool are_different = false;
  if (embedding1.embeddings.size() > 0 && embedding2.embeddings.size() > 0) {
    const auto &vec1 = embedding1.embeddings[0];
    const auto &vec2 = embedding2.embeddings[0];

    if (vec1.size() == vec2.size()) {
      for (size_t i = 0; i < vec1.size(); ++i) {
        if (std::abs(vec1[i] - vec2[i]) > 1e-6) {
          are_different = true;
          break;
        }
      }
    }
  }

  EXPECT_TRUE(are_different) << "Embeddings for different texts should differ";
}

/**
 * @brief Test embeddings for similar texts are similar.
 */
TEST_F(LlamaEmbeddingsTest, SimilarTextsProduceSimilarEmbeddings) {
  const std::string text1 = "Hello world";
  const std::string text2 = "Hello world"; // Identical text

  auto result1 = llama->generate_embeddings(text1);
  auto result2 = llama->generate_embeddings(text2);

  ASSERT_TRUE(result1.is_ok());
  ASSERT_TRUE(result2.is_ok());

  auto embedding1 = result1.value();
  auto embedding2 = result2.value();

  ASSERT_FALSE(embedding1.embeddings.empty());
  ASSERT_FALSE(embedding2.embeddings.empty());

  // Identical texts should produce identical (or very similar) embeddings
  if (embedding1.embeddings.size() > 0 && embedding2.embeddings.size() > 0) {
    const auto &vec1 = embedding1.embeddings[0];
    const auto &vec2 = embedding2.embeddings[0];

    ASSERT_EQ(vec1.size(), vec2.size());

    // Calculate cosine similarity
    float dot_product = 0.0f;
    float norm1 = 0.0f;
    float norm2 = 0.0f;

    for (size_t i = 0; i < vec1.size(); ++i) {
      dot_product += vec1[i] * vec2[i];
      norm1 += vec1[i] * vec1[i];
      norm2 += vec2[i] * vec2[i];
    }

    float cosine_similarity =
        dot_product / (std::sqrt(norm1) * std::sqrt(norm2));

    // Identical texts should have cosine similarity very close to 1.0
    EXPECT_NEAR(cosine_similarity, 1.0f, 0.01f);
  }
}

/**
 * @brief Test embedding generation with empty text.
 */
TEST_F(LlamaEmbeddingsTest, HandlesEmptyText) {
  const std::string empty_text = "";

  auto result = llama->generate_embeddings(empty_text);

  // Empty text should return an error
  ASSERT_TRUE(result.is_error());
  EXPECT_FALSE(result.error().empty());
}

/**
 * @brief Test embedding generation with long text.
 */
TEST_F(LlamaEmbeddingsTest, HandlesLongText) {
  std::string long_text = "This is a very long text that contains many words. ";
  for (int i = 0; i < 10; ++i) {
    long_text += "More content to make this text longer and test how the "
                 "embedding model handles longer sequences. ";
  }

  auto result = llama->generate_embeddings(long_text);

  ASSERT_TRUE(result.is_ok());
  auto embedding_result = result.value();

  EXPECT_FALSE(embedding_result.embeddings.empty());
  EXPECT_GT(embedding_result.n_tokens, 10);
}
