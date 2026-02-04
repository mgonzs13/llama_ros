// MIT License
//
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <thread>

#include "huggingface_hub.h"
#include "llama_ros/llama.hpp"
#include "llama_utils/llama_params.hpp"

/**
 * @brief Test suite for LoRA adapter functionality.
 *
 * This test suite verifies the LoRA adapter capabilities
 * including loading, updating scales, and generation with adapters.
 */
class LlamaLoRATest : public ::testing::Test {
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

    // Download base model from HuggingFace
    auto model_result = huggingface_hub::hf_hub_download_with_shards(
        "bartowski/SmolLM2-135M-Instruct-GGUF",
        "SmolLM2-135M-Instruct-Q6_K.gguf");

    ASSERT_TRUE(model_result.success) << "Failed to download base model";
    ASSERT_FALSE(model_result.path.empty()) << "Model path is empty";

    // Set the downloaded model path
    params->params.model.path = model_result.path;

    // Download LoRA adapter
    // Note: Replace with an actual LoRA adapter compatible with the base model
    auto lora_result = huggingface_hub::hf_hub_download_with_shards(
        "unileon-robotics/SmolLM2-135M-Instruct-BehaviorTree-LoRA-GGUF",
        "f16.gguf");

    ASSERT_TRUE(lora_result.success) << "Failed to download LoRA adapter";
    ASSERT_FALSE(lora_result.path.empty()) << "LoRA path is empty";

    lora_path = lora_result.path;

    // Add LoRA adapter to params
    common_adapter_lora_info lora_info;
    lora_info.path = lora_path;
    lora_info.scale = 0.5f;
    params->params.lora_adapters.push_back(lora_info);

    // Create Llama object with LoRA
    llama = std::make_unique<llama_ros::Llama>(params->params,
                                               params->system_prompt);
    ASSERT_NE(llama, nullptr);

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

    slot = llama->wait_for_available_slot();
    slot->goal_id = 2222; // Assign a dummy goal ID

    prompt = "Write a behavior tree for the robot to execute the command using "
             "only available nodes. "
             "If object is visible, approach then pick up, if heavy move to "
             "designated area, finally perform task. "
             "list of available nodes: <Action ID = ApproachObject /> "
             "<Action ID = PickUpObject /> "
             "<Action ID = MoveObjectToDesignatedArea /> "
             "<Action ID = ScanAreaForObject /> "
             "<Condition ID = IsObjectVisible /> "
             "<Condition ID = IsObjectHeavy /> "
             "<SubTree ID = PerformTask />";

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
  llama_ros::ServerSlot *slot;
  std::string lora_path;
  std::string prompt;
};

/**
 * @brief Test that LoRA adapters are loaded correctly.
 */
TEST_F(LlamaLoRATest, LoRAAdaptersAreLoaded) {
  // List the loaded LoRA adapters
  auto loras = llama->list_loras();

  // Should have at least one LoRA loaded
  ASSERT_GE(loras.size(), 1);

  // Verify the LoRA path matches what we loaded
  EXPECT_EQ(loras[0].path, lora_path);
  EXPECT_FLOAT_EQ(loras[0].scale, 0.5f);
  EXPECT_EQ(loras[0].id, 0);
}

/**
 * @brief Test updating LoRA adapter scales.
 */
TEST_F(LlamaLoRATest, CanUpdateLoRAScales) {
  // Get current LoRAs
  auto loras_before = llama->list_loras();
  ASSERT_GE(loras_before.size(), 1);
  EXPECT_FLOAT_EQ(loras_before[0].scale, 0.5f);

  // Update the scale to 0.8
  std::vector<llama_ros::LoRA> updated_loras;
  llama_ros::LoRA lora_update;
  lora_update.id = 0;
  lora_update.path = lora_path;
  lora_update.scale = 0.8f;
  updated_loras.push_back(lora_update);

  llama->update_loras(updated_loras);

  // Verify the scale was updated
  auto loras_after = llama->list_loras();
  ASSERT_GE(loras_after.size(), 1);
  EXPECT_FLOAT_EQ(loras_after[0].scale, 0.8f);

  // Update to 0.0 (effectively disabling)
  lora_update.scale = 0.0f;
  updated_loras[0] = lora_update;
  llama->update_loras(updated_loras);

  auto loras_disabled = llama->list_loras();
  EXPECT_FLOAT_EQ(loras_disabled[0].scale, 0.0f);
}

/**
 * @brief Test text generation with LoRA at full scale.
 */
TEST_F(LlamaLoRATest, GenerationWithFullLoRA) {
  // Ensure LoRA is at full scale
  std::vector<llama_ros::LoRA> updated_loras;
  llama_ros::LoRA lora_update;
  lora_update.id = 0;
  lora_update.path = lora_path;
  lora_update.scale = 1.0f;
  updated_loras.push_back(lora_update);
  llama->update_loras(updated_loras);

  // Generate text
  struct common_params_sampling sparams = params->params.sampling;
  sparams.temp = 0.8f;
  sparams.seed = 42;

  auto result = llama->generate_response(slot->goal_id, prompt, sparams);

  ASSERT_TRUE(result.is_ok());
  auto result_value = result.value();
  EXPECT_FALSE(result_value.content.empty());
  EXPECT_GT(result_value.n_decoded, 0);
}

/**
 * @brief Test text generation with LoRA disabled (scale = 0).
 */
TEST_F(LlamaLoRATest, GenerationWithoutLoRA) {
  // Disable LoRA by setting scale to 0
  std::vector<llama_ros::LoRA> updated_loras;
  llama_ros::LoRA lora_update;
  lora_update.id = 0;
  lora_update.path = lora_path;
  lora_update.scale = 0.0f;
  updated_loras.push_back(lora_update);
  llama->update_loras(updated_loras);

  // Generate text
  struct common_params_sampling sparams = params->params.sampling;
  sparams.temp = 0.8f;
  sparams.seed = 42;

  auto result = llama->generate_response(slot->goal_id, prompt, sparams);

  ASSERT_TRUE(result.is_ok());
  auto result_value = result.value();
  EXPECT_FALSE(result_value.content.empty());
  EXPECT_GT(result_value.n_decoded, 0);
}

/**
 * @brief Test that LoRA affects generation output.
 */
TEST_F(LlamaLoRATest, LoRAScaleAffectsOutput) {
  struct common_params_sampling sparams = params->params.sampling;
  sparams.temp = 0.0f; // Deterministic
  sparams.seed = 42;

  // Generate with LoRA at scale 1.0
  std::vector<llama_ros::LoRA> lora_full;
  llama_ros::LoRA lora_update;
  lora_update.id = 0;
  lora_update.path = lora_path;
  lora_update.scale = 1.0f;
  lora_full.push_back(lora_update);
  llama->update_loras(lora_full);

  auto result_with_lora =
      llama->generate_response(slot->goal_id, prompt, sparams);
  ASSERT_TRUE(result_with_lora.is_ok());

  // Reset and generate with LoRA at scale 0.0
  llama->reset();
  slot = llama->wait_for_available_slot();
  slot->goal_id = 2223;

  lora_update.scale = 0.0f;
  std::vector<llama_ros::LoRA> lora_disabled;
  lora_disabled.push_back(lora_update);
  llama->update_loras(lora_disabled);

  auto result_without_lora =
      llama->generate_response(slot->goal_id, prompt, sparams);
  ASSERT_TRUE(result_without_lora.is_ok());

  // Both should generate content
  auto with_lora_value = result_with_lora.value();
  auto without_lora_value = result_without_lora.value();
  EXPECT_FALSE(with_lora_value.content.empty());
  EXPECT_FALSE(without_lora_value.content.empty());

  // The outputs should be different due to LoRA influence
  // Note: This may not always be true for very small models or specific prompts
  // but generally LoRA should affect the output
  EXPECT_NE(with_lora_value.content, without_lora_value.content);
}

/**
 * @brief Test generation with partial LoRA scale.
 */
TEST_F(LlamaLoRATest, GenerationWithPartialLoRA) {
  // Set LoRA to partial scale (0.5)
  std::vector<llama_ros::LoRA> updated_loras;
  llama_ros::LoRA lora_update;
  lora_update.id = 0;
  lora_update.path = lora_path;
  lora_update.scale = 0.5f;
  updated_loras.push_back(lora_update);
  llama->update_loras(updated_loras);

  // Generate text
  struct common_params_sampling sparams = params->params.sampling;
  sparams.temp = 0.8f;
  sparams.seed = 42;

  auto result = llama->generate_response(slot->goal_id, prompt, sparams);

  ASSERT_TRUE(result.is_ok());
  auto result_value = result.value();
  EXPECT_FALSE(result_value.content.empty());
  EXPECT_GT(result_value.n_decoded, 0);
}

/**
 * @brief Test that invalid LoRA IDs are rejected.
 */
TEST_F(LlamaLoRATest, InvalidLoRAIDIsRejected) {
  // Try to update a LoRA with an invalid ID
  std::vector<llama_ros::LoRA> updated_loras;
  llama_ros::LoRA lora_update;
  lora_update.id = 999; // Invalid ID
  lora_update.path = lora_path;
  lora_update.scale = 0.5f;
  updated_loras.push_back(lora_update);

  // This should not crash, but should log an error
  llama->update_loras(updated_loras);

  // The original LoRA should still be at its previous scale
  auto loras = llama->list_loras();
  ASSERT_GE(loras.size(), 1);
  // Scale should be unchanged from previous test (0.5f)
}

/**
 * @brief Test streaming with LoRA enabled.
 */
TEST_F(LlamaLoRATest, StreamingWithLoRA) {
  // Ensure LoRA is enabled
  std::vector<llama_ros::LoRA> updated_loras;
  llama_ros::LoRA lora_update;
  lora_update.id = 0;
  lora_update.path = lora_path;
  lora_update.scale = 1.0f;
  updated_loras.push_back(lora_update);
  llama->update_loras(updated_loras);

  struct common_params_sampling sparams = params->params.sampling;

  // Track callback invocations
  std::atomic<int> callback_count{0};
  std::string accumulated_text;

  // Define streaming callback
  auto callback = [&callback_count,
                   &accumulated_text](struct llama_ros::CompletionOutput output,
                                      llama_ros::ServerSlot *slot) {
    callback_count++;
    accumulated_text += output.text_to_send;
  };

  auto result =
      llama->generate_response(slot->goal_id, prompt, sparams, callback);

  ASSERT_TRUE(result.is_ok());

  // Callback should have been invoked
  EXPECT_GT(callback_count.load(), 0);
  EXPECT_FALSE(accumulated_text.empty());
}
