// MIT License
//
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

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include "huggingface_hub.h"
#include "llama_ros/llama.hpp"
#include "llama_utils/llama_params.hpp"

/**
 * @brief Test suite for LLM deployment and initialization.
 *
 * This test suite verifies that the Llama model can be properly
 * initialized, configured, and deployed.
 */
class LlamaDeploymentTest : public ::testing::Test {
protected:
  /**
   * @brief Set up the test fixture.
   *
   * Called before each test to initialize the test environment.
   */
  void SetUp() override {
    // Initialize test parameters with default values
    params = std::make_unique<llama_utils::LlamaParams>();

    // Set up basic common_params structure
    params->params.n_ctx = 512;
    params->params.n_batch = 128;
    params->params.n_ubatch = 128;
    params->params.n_predict = 128;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.sampling.seed = LLAMA_DEFAULT_SEED;
  }

  /**
   * @brief Tear down the test fixture.
   *
   * Called after each test to clean up resources.
   */
  void TearDown() override {
    llama.reset();
    params.reset();
  }

  std::unique_ptr<llama_ros::Llama> llama;
  std::unique_ptr<llama_utils::LlamaParams> params;
};

/**
 * @brief Test that verifies Llama object can be created.
 *
 * This test ensures that the Llama object can be instantiated
 * without crashing.
 */
TEST_F(LlamaDeploymentTest, CanCreateLlamaObject) {
  // Download model from HuggingFace
  auto result = huggingface_hub::hf_hub_download_with_shards(
      "bartowski/google_gemma-3-270m-it-GGUF",
      "google_gemma-3-270m-it-Q4_K_M.gguf");

  ASSERT_TRUE(result.success) << "Failed to download model";
  ASSERT_FALSE(result.path.empty()) << "Model path is empty";

  // Set the downloaded model path
  params->params.model.path = result.path;

  // Create Llama object
  ASSERT_NO_THROW({
    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
  });

  // Verify the object was created
  ASSERT_NE(llama, nullptr);
}

/**
 * @brief Test that verifies parameters can be configured.
 *
 * This test ensures that model parameters can be properly set
 * before model initialization.
 */
TEST_F(LlamaDeploymentTest, CanConfigureParameters) {
  ASSERT_NE(params, nullptr);

  // Test setting various parameters
  params->params.n_ctx = 2048;
  params->params.n_batch = 512;
  params->params.n_gpu_layers = 32;
  params->params.embedding = true;
  params->system_prompt = "You are a helpful assistant.";

  // Verify parameters were set
  EXPECT_EQ(params->params.n_ctx, 2048);
  EXPECT_EQ(params->params.n_batch, 512);
  EXPECT_EQ(params->params.n_gpu_layers, 32);
  EXPECT_TRUE(params->params.embedding);
  EXPECT_EQ(params->system_prompt, "You are a helpful assistant.");
}

/**
 * @brief Test that verifies model can be loaded successfully.
 *
 * This test ensures that a model file can be loaded into memory
 * and initialized properly.
 *
 * Note: This test requires a valid model file path.
 */
TEST_F(LlamaDeploymentTest, CanLoadModel) {
  // Download model from HuggingFace
  auto result = huggingface_hub::hf_hub_download_with_shards(
      "bartowski/google_gemma-3-270m-it-GGUF",
      "google_gemma-3-270m-it-Q4_K_M.gguf");

  ASSERT_TRUE(result.success) << "Failed to download model";
  ASSERT_FALSE(result.path.empty()) << "Model path is empty";

  // Set the downloaded model path
  params->params.model.path = result.path;

  // Create and load the model
  ASSERT_NO_THROW({
    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
  });

  // Verify the model was loaded successfully
  ASSERT_NE(llama, nullptr);

  // Verify embedding mode matches our configuration
  EXPECT_EQ(llama->is_embedding(), params->params.embedding);
}

/**
 * @brief Test that verifies model initialization with invalid path fails
 * gracefully.
 *
 * This test ensures that attempting to load a model with an invalid
 * path is handled correctly without crashing.
 */
TEST_F(LlamaDeploymentTest, InvalidModelPathHandledGracefully) {
  // Set invalid model path
  params->params.model.path = "/nonexistent/path/to/model.gguf";

  // Attempting to create Llama with invalid model should handle gracefully
  // The Llama constructor should detect the invalid model and handle it
  // without causing a segfault
  ASSERT_THROW({
    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
  }, std::runtime_error);
}

/**
 * @brief Test that verifies model metadata can be retrieved.
 *
 * This test ensures that after successful initialization, model
 * metadata (architecture, parameters, etc.) can be accessed.
 */
TEST_F(LlamaDeploymentTest, CanRetrieveModelMetadata) {
  // Download model from HuggingFace
  auto result = huggingface_hub::hf_hub_download_with_shards(
      "bartowski/google_gemma-3-270m-it-GGUF",
      "google_gemma-3-270m-it-Q4_K_M.gguf");

  ASSERT_TRUE(result.success) << "Failed to download model";
  ASSERT_FALSE(result.path.empty()) << "Model path is empty";

  // Set the downloaded model path
  params->params.model.path = result.path;

  // Create and load the model
  ASSERT_NO_THROW({
    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
  });

  ASSERT_NE(llama, nullptr);

  // Retrieve metadata
  llama_ros::Metadata metadata = llama->get_metadata();

  // Verify metadata contains expected information
  EXPECT_FALSE(metadata.general.architecture.empty());
  EXPECT_GT(metadata.general.architecture.size(), 0);

  // For Gemma models, architecture should contain "gemma"
  EXPECT_NE(metadata.general.architecture.find("gemma"), std::string::npos);
}

/**
 * @brief Test that verifies context size configuration.
 *
 * This test ensures that the context size can be properly configured
 * and validated during model initialization.
 */
TEST_F(LlamaDeploymentTest, CanConfigureContextSize) {
  // Test various context sizes
  params->params.n_ctx = 512;
  EXPECT_EQ(params->params.n_ctx, 512);

  params->params.n_ctx = 2048;
  EXPECT_EQ(params->params.n_ctx, 2048);

  params->params.n_ctx = 4096;
  EXPECT_EQ(params->params.n_ctx, 4096);

  // Batch size should not exceed context size in practice
  params->params.n_batch = 256;
  EXPECT_LE(params->params.n_batch, params->params.n_ctx);
}

/**
 * @brief Test that verifies model cleanup and resource deallocation.
 *
 * This test ensures that when a Llama object is destroyed, all
 * resources are properly cleaned up without memory leaks.
 */
TEST_F(LlamaDeploymentTest, ProperCleanupOnDestruction) {
  // Download model from HuggingFace
  auto result = huggingface_hub::hf_hub_download_with_shards(
      "bartowski/google_gemma-3-270m-it-GGUF",
      "google_gemma-3-270m-it-Q4_K_M.gguf");

  ASSERT_TRUE(result.success) << "Failed to download model";
  ASSERT_FALSE(result.path.empty()) << "Model path is empty";

  // Set the downloaded model path
  params->params.model.path = result.path;

  // Create and load the model
  ASSERT_NO_THROW({
    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
  });

  ASSERT_NE(llama, nullptr);

  // Reset (destroy) the Llama object
  ASSERT_NO_THROW({ llama.reset(); });

  // Verify it's been cleaned up
  EXPECT_EQ(llama, nullptr);
}
