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

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

#include "llama_utils/llama_params.hpp"

/**
 * @brief Test suite for LLM parameter configuration.
 *
 * This test suite verifies that the parameter loading and configuration
 * functions work correctly.
 */
class LlamaParamsTest : public ::testing::Test {
protected:
  void SetUp() override {
    rclcpp::init(0, nullptr);

    // Create a lifecycle node for testing
    node = std::make_shared<rclcpp_lifecycle::LifecycleNode>(
        "test_llama_params_node");

    // Declare the llama parameters
    llama_utils::declare_llama_params(node);
  }

  void TearDown() override {
    node.reset();
    rclcpp::shutdown();
  }

  rclcpp_lifecycle::LifecycleNode::SharedPtr node;
};

/**
 * @brief Test that verifies parameters can be declared.
 */
TEST_F(LlamaParamsTest, CanDeclareParameters) {
  ASSERT_NE(node, nullptr);

  // Verify some key parameters exist
  ASSERT_TRUE(node->has_parameter("n_ctx"));
  ASSERT_TRUE(node->has_parameter("n_batch"));
  ASSERT_TRUE(node->has_parameter("n_gpu_layers"));
  ASSERT_TRUE(node->has_parameter("seed"));
  ASSERT_TRUE(node->has_parameter("model_path"));
  ASSERT_TRUE(node->has_parameter("embedding"));
}

/**
 * @brief Test that verifies default parameter values.
 */
TEST_F(LlamaParamsTest, DefaultParameterValues) {
  int32_t n_ctx = node->get_parameter("n_ctx").as_int();
  int32_t n_batch = node->get_parameter("n_batch").as_int();
  int32_t seed = node->get_parameter("seed").as_int();
  bool embedding = node->get_parameter("embedding").as_bool();

  EXPECT_EQ(n_ctx, 0);
  EXPECT_EQ(n_batch, 2048);
  EXPECT_EQ(seed, -1);
  EXPECT_FALSE(embedding);
}

/**
 * @brief Test that verifies parameters can be set.
 */
TEST_F(LlamaParamsTest, CanSetParameters) {
  // Set custom values
  node->set_parameter(rclcpp::Parameter("n_ctx", 1024));
  node->set_parameter(rclcpp::Parameter("n_batch", 512));
  node->set_parameter(rclcpp::Parameter("seed", 42));
  node->set_parameter(rclcpp::Parameter("embedding", true));

  // Verify they were set
  EXPECT_EQ(node->get_parameter("n_ctx").as_int(), 1024);
  EXPECT_EQ(node->get_parameter("n_batch").as_int(), 512);
  EXPECT_EQ(node->get_parameter("seed").as_int(), 42);
  EXPECT_TRUE(node->get_parameter("embedding").as_bool());
}

/**
 * @brief Test that verifies get_llama_params returns valid structure.
 */
TEST_F(LlamaParamsTest, GetLlamaParamsReturnsValidStruct) {
  // Set some test parameters
  node->set_parameter(rclcpp::Parameter("n_ctx", 2048));
  node->set_parameter(rclcpp::Parameter("n_batch", 1024));
  node->set_parameter(rclcpp::Parameter("n_gpu_layers", 32));
  node->set_parameter(rclcpp::Parameter("seed", 12345));
  node->set_parameter(rclcpp::Parameter("model_path", "/path/to/model.gguf"));

  // Get the params structure
  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  // Verify the values were correctly loaded
  EXPECT_EQ(params.params.n_ctx, 2048);
  EXPECT_EQ(params.params.n_batch, 1024);
  EXPECT_EQ(params.params.n_gpu_layers, 32);
  EXPECT_EQ(params.params.sampling.seed, 12345);
  EXPECT_EQ(params.params.model.path, "/path/to/model.gguf");
}

/**
 * @brief Test that verifies seed handling (negative seed becomes default).
 */
TEST_F(LlamaParamsTest, NegativeSeedBecomesDefault) {
  node->set_parameter(rclcpp::Parameter("seed", -1));

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  // Negative seed should be converted to LLAMA_DEFAULT_SEED
  EXPECT_EQ(params.params.sampling.seed, LLAMA_DEFAULT_SEED);
}

/**
 * @brief Test that verifies thread count defaults.
 */
TEST_F(LlamaParamsTest, ThreadCountDefaults) {
  node->set_parameter(rclcpp::Parameter("n_threads", -1));

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  // Negative thread counts should be set to cpu_get_num_math()
  EXPECT_GT(params.params.cpuparams.n_threads, 0);
  EXPECT_GT(params.params.cpuparams_batch.n_threads, 0);
}

/**
 * @brief Test that verifies LoRA adapter configuration.
 */
TEST_F(LlamaParamsTest, LoRAAdapterConfiguration) {
  std::vector<std::string> loras = {"adapter1", "adapter2"};

  node->set_parameter(rclcpp::Parameter("loras", loras));

  // Declare and set per-lora parameters (simulating what get_llama_params does
  // internally)
  node->declare_parameter<std::string>("adapter1.file_path",
                                       "/path/to/adapter1.bin");
  node->declare_parameter<double>("adapter1.scale", 0.5);
  node->declare_parameter<std::string>("adapter2.file_path",
                                       "/path/to/adapter2.bin");
  node->declare_parameter<double>("adapter2.scale", 0.8);

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  // Should have 2 adapters configured
  EXPECT_EQ(params.params.lora_adapters.size(), 2);
  EXPECT_EQ(params.params.lora_adapters[0].scale, 0.5f);
  EXPECT_EQ(params.params.lora_adapters[1].scale, 0.8f);
}

/**
 * @brief Test that verifies LoRA scale clamping (0.0 to 1.0).
 */
TEST_F(LlamaParamsTest, LoRAScaleClamping) {
  std::vector<std::string> loras = {"adapter1", "adapter2"};

  node->set_parameter(rclcpp::Parameter("loras", loras));

  // Declare per-lora parameters with invalid scales
  node->declare_parameter<std::string>("adapter1.file_path",
                                       "/path/to/adapter1.bin");
  node->declare_parameter<double>("adapter1.scale", -0.5);
  node->declare_parameter<std::string>("adapter2.file_path",
                                       "/path/to/adapter2.bin");
  node->declare_parameter<double>("adapter2.scale", 1.5);

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  // Scales should be clamped to [0.0, 1.0]
  EXPECT_EQ(params.params.lora_adapters[0].scale, 0.0f);
  EXPECT_EQ(params.params.lora_adapters[1].scale, 1.0f);
}

/**
 * @brief Test that verifies stopping words configuration.
 */
TEST_F(LlamaParamsTest, StoppingWordsConfiguration) {
  std::vector<std::string> stop_words = {"STOP", "END", "\\n\\n"};

  node->set_parameter(rclcpp::Parameter("stopping_words", stop_words));

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  // Should have 3 antiprompts (stopping words)
  EXPECT_EQ(params.params.antiprompt.size(), 3);
  EXPECT_EQ(params.params.antiprompt[0], "STOP");
  EXPECT_EQ(params.params.antiprompt[1], "END");
  // \\n should be converted to actual newline
  EXPECT_EQ(params.params.antiprompt[2], "\n\n");
}

/**
 * @brief Test that verifies split mode configuration.
 */
TEST_F(LlamaParamsTest, SplitModeConfiguration) {
  // Test "layer" mode
  node->set_parameter(rclcpp::Parameter("split_mode", "layer"));
  llama_utils::LlamaParams params1 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params1.params.split_mode, LLAMA_SPLIT_MODE_LAYER);

  // Test "none" mode
  node->set_parameter(rclcpp::Parameter("split_mode", "none"));
  llama_utils::LlamaParams params2 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params2.params.split_mode, LLAMA_SPLIT_MODE_NONE);

  // Test "row" mode
  node->set_parameter(rclcpp::Parameter("split_mode", "row"));
  llama_utils::LlamaParams params3 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params3.params.split_mode, LLAMA_SPLIT_MODE_ROW);
}

/**
 * @brief Test that verifies RoPE scaling type configuration.
 */
TEST_F(LlamaParamsTest, RoPEScalingTypeConfiguration) {
  // Test "linear" scaling
  node->set_parameter(rclcpp::Parameter("rope_scaling_type", "linear"));
  llama_utils::LlamaParams params1 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params1.params.rope_scaling_type, LLAMA_ROPE_SCALING_TYPE_LINEAR);

  // Test "yarn" scaling
  node->set_parameter(rclcpp::Parameter("rope_scaling_type", "yarn"));
  llama_utils::LlamaParams params2 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params2.params.rope_scaling_type, LLAMA_ROPE_SCALING_TYPE_YARN);

  // Test "none" scaling
  node->set_parameter(rclcpp::Parameter("rope_scaling_type", "none"));
  llama_utils::LlamaParams params3 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params3.params.rope_scaling_type, LLAMA_ROPE_SCALING_TYPE_NONE);
}

/**
 * @brief Test that verifies pooling type configuration.
 */
TEST_F(LlamaParamsTest, PoolingTypeConfiguration) {
  // Test "mean" pooling
  node->set_parameter(rclcpp::Parameter("pooling_type", "mean"));
  llama_utils::LlamaParams params1 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params1.params.pooling_type, LLAMA_POOLING_TYPE_MEAN);

  // Test "cls" pooling
  node->set_parameter(rclcpp::Parameter("pooling_type", "cls"));
  llama_utils::LlamaParams params2 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params2.params.pooling_type, LLAMA_POOLING_TYPE_CLS);

  // Test reranking mode (should set pooling to RANK and embedding to true)
  node->set_parameter(rclcpp::Parameter("reranking", true));
  llama_utils::LlamaParams params3 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params3.params.pooling_type, LLAMA_POOLING_TYPE_RANK);
  EXPECT_TRUE(params3.params.embedding);
}

/**
 * @brief Test that verifies NUMA configuration.
 */
TEST_F(LlamaParamsTest, NUMAConfiguration) {
  // Test "distribute" strategy
  node->set_parameter(rclcpp::Parameter("numa", "distribute"));
  llama_utils::LlamaParams params1 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params1.params.numa, GGML_NUMA_STRATEGY_DISTRIBUTE);

  // Test "isolate" strategy
  node->set_parameter(rclcpp::Parameter("numa", "isolate"));
  llama_utils::LlamaParams params2 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params2.params.numa, GGML_NUMA_STRATEGY_ISOLATE);

  // Test "none" strategy
  node->set_parameter(rclcpp::Parameter("numa", "none"));
  llama_utils::LlamaParams params3 = llama_utils::get_llama_params(node);
  EXPECT_EQ(params3.params.numa, GGML_NUMA_STRATEGY_DISABLED);
}

/**
 * @brief Test that verifies system prompt configuration.
 */
TEST_F(LlamaParamsTest, SystemPromptConfiguration) {
  std::string test_prompt = "You are a helpful assistant.";
  node->set_parameter(rclcpp::Parameter("system_prompt", test_prompt));

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  EXPECT_EQ(params.system_prompt, test_prompt);
}

/**
 * @brief Test that verifies prefix and suffix configuration.
 */
TEST_F(LlamaParamsTest, PrefixSuffixConfiguration) {
  node->set_parameter(rclcpp::Parameter("prefix", "### Instruction:\n"));
  node->set_parameter(rclcpp::Parameter("suffix", "\n### Response:\n"));

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  EXPECT_EQ(params.params.input_prefix, "### Instruction:\n");
  EXPECT_EQ(params.params.input_suffix, "\n### Response:\n");
}

/**
 * @brief Test that verifies parse_priority function.
 */
TEST_F(LlamaParamsTest, ParsePriorityFunction) {
  EXPECT_EQ(llama_utils::parse_priority("normal"), GGML_SCHED_PRIO_NORMAL);
  EXPECT_EQ(llama_utils::parse_priority("medium"), GGML_SCHED_PRIO_MEDIUM);
  EXPECT_EQ(llama_utils::parse_priority("high"), GGML_SCHED_PRIO_HIGH);
  EXPECT_EQ(llama_utils::parse_priority("realtime"), GGML_SCHED_PRIO_REALTIME);
  EXPECT_EQ(llama_utils::parse_priority("invalid"), GGML_SCHED_PRIO_NORMAL);
}

/**
 * @brief Test that verifies CPU priority configuration.
 */
TEST_F(LlamaParamsTest, CPUPriorityConfiguration) {
  node->set_parameter(rclcpp::Parameter("priority", "high"));
  node->set_parameter(rclcpp::Parameter("priority_batch", "medium"));

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  EXPECT_EQ(params.params.cpuparams.priority, GGML_SCHED_PRIO_HIGH);
  EXPECT_EQ(params.params.cpuparams_batch.priority, GGML_SCHED_PRIO_MEDIUM);
}

/**
 * @brief Test that verifies parallel processing configuration.
 */
TEST_F(LlamaParamsTest, ParallelProcessingConfiguration) {
  node->set_parameter(rclcpp::Parameter("n_parallel", 4));
  node->set_parameter(rclcpp::Parameter("n_sequences", 2));
  node->set_parameter(rclcpp::Parameter("cont_batching", false));

  llama_utils::LlamaParams params = llama_utils::get_llama_params(node);

  EXPECT_EQ(params.params.n_parallel, 4);
  EXPECT_EQ(params.params.n_sequences, 2);
  EXPECT_FALSE(params.params.cont_batching);
}
