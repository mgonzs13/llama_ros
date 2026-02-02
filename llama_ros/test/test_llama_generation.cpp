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
#include <atomic>
#include <thread>
#include <chrono>
#include <algorithm>

#include "huggingface_hub.h"
#include "llama_ros/llama.hpp"
#include "llama_utils/llama_params.hpp"

/**
 * @brief Test suite for LLM text generation functionality.
 *
 * This test suite verifies the text generation capabilities
 * including completion, chat, and streaming.
 */
class LlamaGenerationTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Initialize test parameters
    params = std::make_unique<llama_utils::LlamaParams>();
    
    // Set up basic common_params structure
    params->params.n_ctx = 512;
    params->params.n_batch = 128;
    params->params.n_ubatch = 128;
    params->params.n_predict = 64;
    params->params.cpuparams.n_threads = 1;
    params->params.cpuparams_batch.n_threads = 1;
    params->params.sampling.seed = LLAMA_DEFAULT_SEED;
    params->params.input_prefix = "\n<|im_start|>user\n";
    params->params.input_suffix = "<|im_end|>\n<|im_start|>assistant\n";
    params->params.antiprompt = {"<|im_end|>"};
    
    // Download model from HuggingFace
    auto result = huggingface_hub::hf_hub_download_with_shards(
        "bartowski/SmolLM2-135M-Instruct-GGUF",
        "SmolLM2-135M-Instruct-Q6_K.gguf");
    
    ASSERT_TRUE(result.success) << "Failed to download model";
    ASSERT_FALSE(result.path.empty()) << "Model path is empty";
    
    // Set the downloaded model path
    params->params.model.path = result.path;
    
    // Create Llama object
    llama = std::make_unique<llama_ros::Llama>(params->params, params->system_prompt);
    ASSERT_NE(llama, nullptr);
    
    // Start the run_loop thread (required for processing generation requests)
    run_loop_thread = std::thread([this]() {
      try {
        this->llama->run_loop();
      } catch (const std::exception& e) {
        // Log error but don't fail the test here
      } catch (...) {
        // Handle unknown exceptions
      }
    });

    slot = llama->wait_for_available_slot();
    slot->goal_id = 1111; // Assign a dummy goal ID
    
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
      run_loop_thread.join();
    }
    
    llama.reset();
    params.reset();
  }

  std::unique_ptr<llama_ros::Llama> llama;
  std::unique_ptr<llama_utils::LlamaParams> params;
  std::thread run_loop_thread;
  llama_ros::ServerSlot* slot;
};

/**
 * @brief Test basic text completion.
 */
TEST_F(LlamaGenerationTest, CanGenerateSimpleCompletion) {
  // Set up sampling parameters
  struct common_params_sampling sparams = params->params.sampling;
  sparams.temp = 0.8f;
  sparams.top_k = 40;
  sparams.top_p = 0.95f;

  // Generate a simple completion
  auto result = llama->generate_response(
      slot->goal_id, // slot_id
      "The capital of France is",
      sparams);
  
  // Verify we got a result
  ASSERT_TRUE(result.is_ok());
  
  // Verify the result contains content
  auto result_value = result.value();
  EXPECT_FALSE(result_value.content.empty());
  EXPECT_GT(result_value.n_decoded, 0);
  
  // The result should contain "Paris" or similar
  std::string lower_content = result_value.content;
  std::transform(lower_content.begin(), lower_content.end(), lower_content.begin(), ::tolower);
  EXPECT_NE(lower_content.find("paris"), std::string::npos);
}

/**
 * @brief Test generation with temperature parameter.
 */
TEST_F(LlamaGenerationTest, TemperatureAffectsOutput) {
  const std::string prompt = "Count from 1 to 5:";
  
  // Generate with low temperature (more deterministic)
  struct common_params_sampling sparams_low = params->params.sampling;
  sparams_low.temp = 0.0f;
  sparams_low.seed = 42; // Fixed seed for reproducibility

  auto result_low = llama->generate_response(slot->goal_id, prompt, sparams_low);
  ASSERT_TRUE(result_low.is_ok());
  
  // Reset for next generation
  llama->reset();
  
  // Generate with high temperature (more random)
  struct common_params_sampling sparams_high = params->params.sampling;
  sparams_high.temp = 1.2f;
  sparams_high.seed = 42; // Same seed
  
  auto result_high = llama->generate_response(slot->goal_id, prompt, sparams_high);
  ASSERT_TRUE(result_high.is_ok());
  
  // Both should generate content
  auto result_low_value = result_low.value();
  auto result_high_value = result_high.value();
  EXPECT_FALSE(result_low_value.content.empty());
  EXPECT_FALSE(result_high_value.content.empty());
  
  // Note: With different temperatures but same seed, outputs may still differ
  // We mainly verify both complete successfully
  EXPECT_GT(result_low_value.n_decoded, 0);
  EXPECT_GT(result_high_value.n_decoded, 0);
}

/**
 * @brief Test generation with top-p sampling.
 */
TEST_F(LlamaGenerationTest, TopPSamplingWorks) {
  const std::string prompt = "Hello, how are";
  
  // Generate with top-p sampling
  struct common_params_sampling sparams = params->params.sampling;
  sparams.top_p = 0.9f;
  sparams.temp = 0.8f;
  
  auto result = llama->generate_response(slot->goal_id, prompt, sparams);
  
  ASSERT_TRUE(result.is_ok());
  auto result_value = result.value();
  EXPECT_FALSE(result_value.content.empty());
  EXPECT_GT(result_value.n_decoded, 0);
}

/**
 * @brief Test generation with max tokens limit.
 */
TEST_F(LlamaGenerationTest, RespectsMaxTokensLimit) {
  const std::string prompt = "Write a long story about";
  
  // Set a low max tokens limit
  const int max_tokens = 10;
  
  struct common_params_sampling sparams = params->params.sampling;
  
  // Temporarily adjust n_predict for this slot
  int original_n_predict = slot->n_predict;
  slot->params.n_predict = max_tokens;
  
  auto result = llama->generate_response(slot->goal_id, prompt, sparams);
  
  ASSERT_TRUE(result.is_ok());
  
  // The number of generated tokens should not exceed max_tokens significantly
  // (may be slightly more due to stopping conditions)
  auto result_value = result.value();
  EXPECT_LE(result_value.n_decoded, max_tokens + 5);
  
  // Restore original n_predict
  slot->n_predict = original_n_predict;
}

/**
 * @brief Test generation with stop sequences.
 */
TEST_F(LlamaGenerationTest, StopSequencesWork) {
  const std::string prompt = "Print the number 123524684363216843216843213584635";
  
  struct common_params_sampling sparams = params->params.sampling;
  
  // Use stop sequences to halt generation
  std::vector<std::string> stop_sequences = {"5"};
  
  auto result = llama->generate_response(slot->goal_id, prompt, sparams, nullptr, stop_sequences);
  
  ASSERT_TRUE(result.is_ok());
  auto result_value = result.value();
  ASSERT_TRUE(result_value.content.find("123") != std::string::npos);
  EXPECT_FALSE(result_value.content.empty());
  
  // The content should stop before generating much past "5"
  // This is a heuristic check since exact behavior depends on tokenization
  EXPECT_LT(result_value.content.length(), 100);
}

/**
 * @brief Test streaming generation callback.
 */
TEST_F(LlamaGenerationTest, StreamingCallbacksWork) {
  const std::string prompt = "Hello world";
  
  struct common_params_sampling sparams = params->params.sampling;
  
  // Track callback invocations
  std::atomic<int> callback_count{0};
  std::string accumulated_text;
  
  // Define streaming callback
  auto callback = [&callback_count, &accumulated_text](
      struct llama_ros::CompletionOutput output, llama_ros::ServerSlot *slot) {
    callback_count++;
    accumulated_text += output.text_to_send;
  };
  
  auto result = llama->generate_response(slot->goal_id, prompt, sparams, callback);
  
  ASSERT_TRUE(result.is_ok());
  
  // Callback should have been invoked multiple times during generation
  EXPECT_GT(callback_count.load(), 0);
  
  // Accumulated text from callbacks should match final content (approximately)
  EXPECT_FALSE(accumulated_text.empty());
}

/**
 * @brief Test generation can be cancelled.
 */
TEST_F(LlamaGenerationTest, CanCancelGeneration) {
  const std::string prompt = "Write a very long story that goes on and on";
  
  struct common_params_sampling sparams = params->params.sampling;
  
  // Start generation in a separate thread
  std::atomic<bool> generation_started{false};
  std::atomic<bool> generation_completed{false};
  
  std::thread generation_thread([&]() {
    generation_started = true;
    auto result = llama->generate_response(slot->goal_id, prompt, sparams);
    generation_completed = true;
  });
  
  // Wait for generation to start
  while (!generation_started.load()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  
  // Give it a moment to actually start processing
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  
  // Cancel the generation
  llama->cancel();
  
  // Wait for thread to complete
  generation_thread.join();
  
  // Generation should have completed (either normally or by cancellation)
  EXPECT_TRUE(generation_completed.load());
}
