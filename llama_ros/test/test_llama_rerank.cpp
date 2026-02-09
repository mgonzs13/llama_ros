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
 * @brief Test suite for reranking functionality.
 *
 * This test suite verifies the document reranking capabilities
 * of the LLM, which requires the model to be loaded in reranking mode.
 */
class LlamaRerankingTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize test parameters
    params = std::make_unique<llama_utils::LlamaParams>();

    // Set up basic common_params structure for reranking
    params->params.n_ctx = 512;
    params->params.n_batch = 128;
    params->params.n_ubatch = 128;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.sampling.seed = LLAMA_DEFAULT_SEED;

    // Enable reranking mode
    params->params.embedding = true;
    params->params.pooling_type = LLAMA_POOLING_TYPE_RANK;

    // Download reranking model from HuggingFace
    auto result = huggingface_hub::hf_hub_download_with_shards(
        "gpustack/bge-reranker-v2-m3-GGUF", "bge-reranker-v2-m3-Q4_K_M.gguf");

    ASSERT_TRUE(result.success) << "Failed to download model";
    ASSERT_FALSE(result.path.empty()) << "Model path is empty";

    // Set the downloaded model path
    params->params.model.path = result.path;

    // Create Llama object in reranking mode
    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
    ASSERT_NE(llama, nullptr);
    ASSERT_TRUE(llama->is_reranking());

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
 * @brief Test basic document reranking.
 */
TEST_F(LlamaRerankingTest, CanRerankDocuments) {
  const std::string query = "What is the capital of France?";
  const std::vector<std::string> documents = {
      "Paris is the capital of France.", "London is the capital of England.",
      "Berlin is the capital of Germany."};

  auto result = llama->rank_documents(query, documents);

  ASSERT_TRUE(result.is_ok())
      << "Failed to rerank documents: " << result.error();

  auto reranks = result.value();

  // Should have one score per document
  EXPECT_EQ(reranks.size(), documents.size());

  // Each result should have a valid score
  for (const auto &rerank : reranks) {
    EXPECT_GT(rerank.score, -1e6) << "Score should be initialized";
  }
}

/**
 * @brief Test that relevant document gets higher score.
 */
TEST_F(LlamaRerankingTest, RelevantDocumentScoresHigher) {
  const std::string query = "Information about Python programming";
  const std::vector<std::string> documents = {
      "Python is a high-level programming language.",
      "The snake slithered through the grass.",
      "Java is another programming language."};

  auto result = llama->rank_documents(query, documents);

  ASSERT_TRUE(result.is_ok());
  auto reranks = result.value();

  ASSERT_EQ(reranks.size(), 3);

  // First document about Python programming should score higher than the snake
  // one
  EXPECT_GT(reranks[0].score, reranks[1].score)
      << "Programming-related doc should score higher than unrelated doc";
}

/**
 * @brief Test reranking with single document.
 */
TEST_F(LlamaRerankingTest, HandlesSingleDocument) {
  const std::string query = "Machine learning";
  const std::vector<std::string> documents = {
      "Machine learning is a subset of artificial intelligence."};

  auto result = llama->rank_documents(query, documents);

  ASSERT_TRUE(result.is_ok());
  auto reranks = result.value();

  EXPECT_EQ(reranks.size(), 1);
  EXPECT_GT(reranks[0].score, -1e6);
}

/**
 * @brief Test reranking with many documents.
 */
TEST_F(LlamaRerankingTest, HandlesMultipleDocuments) {
  const std::string query = "Climate change effects";
  std::vector<std::string> documents;

  // Create 5 documents
  documents.push_back("Climate change is causing global temperatures to rise.");
  documents.push_back("The stock market fluctuates daily.");
  documents.push_back("Rising sea levels are a result of climate change.");
  documents.push_back("Football is a popular sport worldwide.");
  documents.push_back("Climate change impacts weather patterns globally.");

  auto result = llama->rank_documents(query, documents);

  ASSERT_TRUE(result.is_ok());
  auto reranks = result.value();

  EXPECT_EQ(reranks.size(), 5);

  // Verify all scores are valid
  for (const auto &rerank : reranks) {
    EXPECT_GT(rerank.score, -1e6);
  }

  // Climate-related docs (0, 2, 4) should generally score higher than unrelated
  // ones
  float climate_avg =
      (reranks[0].score + reranks[2].score + reranks[4].score) / 3.0f;
  float unrelated_avg = (reranks[1].score + reranks[3].score) / 2.0f;

  EXPECT_GT(climate_avg, unrelated_avg)
      << "Relevant documents should score higher on average";
}

/**
 * @brief Test reranking preserves document order in results.
 */
TEST_F(LlamaRerankingTest, PreservesDocumentOrder) {
  const std::string query = "Technology";
  const std::vector<std::string> documents = {
      "First document about computers.", "Second document about smartphones.",
      "Third document about tablets."};

  auto result = llama->rank_documents(query, documents);

  ASSERT_TRUE(result.is_ok());
  auto reranks = result.value();

  ASSERT_EQ(reranks.size(), 3);

  // Results should be in the same order as input (indexed 0, 1, 2)
  // The id field contains the document index in the lower 32 bits
  EXPECT_EQ(reranks[0].id & 0xFFFFFFFF, 0);
  EXPECT_EQ(reranks[1].id & 0xFFFFFFFF, 1);
  EXPECT_EQ(reranks[2].id & 0xFFFFFFFF, 2);
}

/**
 * @brief Test reranking with empty query.
 */
TEST_F(LlamaRerankingTest, HandlesEmptyQuery) {
  const std::string empty_query = "";
  const std::vector<std::string> documents = {"Some document text."};

  auto result = llama->rank_documents(empty_query, documents);

  // Should handle empty query (may return error or succeed with low scores)
  // If it succeeds, scores should be valid
  if (result.is_ok()) {
    auto reranks = result.value();
    EXPECT_EQ(reranks.size(), documents.size());
  }
}

/**
 * @brief Test reranking with empty document.
 */
TEST_F(LlamaRerankingTest, HandlesEmptyDocument) {
  const std::string query = "Search query";
  const std::vector<std::string> documents = {"Valid document",
                                              "", // Empty document
                                              "Another valid document"};

  auto result = llama->rank_documents(query, documents);

  // Should handle empty document in the list
  if (result.is_ok()) {
    auto reranks = result.value();
    EXPECT_EQ(reranks.size(), 3);
  }
}

/**
 * @brief Test parallel processing with n_parallel parameter.
 *
 * This is a standalone test (not using the fixture) to avoid loading
 * the default model in SetUp(), saving memory.
 */
TEST(LlamaRerankingStandaloneTest, ParallelProcessing) {
  // Create a new params with n_parallel = 3 for parallel processing
  auto parallel_params = std::make_unique<llama_utils::LlamaParams>();

  parallel_params->params.n_ctx = 512;
  parallel_params->params.n_batch = 128;
  parallel_params->params.n_ubatch = 128;
  parallel_params->params.cpuparams.n_threads = 1;
  parallel_params->params.cpuparams_batch.n_threads = 1;
  parallel_params->params.sampling.seed = LLAMA_DEFAULT_SEED;
  parallel_params->params.embedding = true;
  parallel_params->params.pooling_type = LLAMA_POOLING_TYPE_RANK;

  // Set n_parallel to 3 to allow processing 3 documents in parallel
  parallel_params->params.n_parallel = 3;

  // Download reranking model
  auto result = huggingface_hub::hf_hub_download_with_shards(
      "gpustack/bge-reranker-v2-m3-GGUF", "bge-reranker-v2-m3-Q4_K_M.gguf");

  ASSERT_TRUE(result.success);
  parallel_params->params.model.path = result.path;

  // Create Llama with parallel processing
  auto parallel_llama = std::make_unique<llama_ros::Llama>(
      parallel_params->params, parallel_params->system_prompt);
  ASSERT_NE(parallel_llama, nullptr);

  // Start run_loop
  std::thread parallel_thread([&parallel_llama]() {
    try {
      parallel_llama->run_loop();
    } catch (...) {
    }
  });
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Test with exactly 3 documents (matching n_parallel)
  const std::string query = "Artificial intelligence applications";
  const std::vector<std::string> documents = {
      "Machine learning is a subset of AI used for predictions.",
      "Deep learning uses neural networks for pattern recognition.",
      "Natural language processing enables computers to understand text."};

  auto rerank_result = parallel_llama->rank_documents(query, documents);

  ASSERT_TRUE(rerank_result.is_ok());
  auto reranks = rerank_result.value();

  EXPECT_EQ(reranks.size(), 3);
  for (const auto &rerank : reranks) {
    EXPECT_GT(rerank.score, -1e6);
  }

  // Cleanup
  parallel_llama->cancel();
  if (parallel_thread.joinable()) {
    parallel_thread.join();
  }
}
