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

#include <atomic>
#include <cstring>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <thread>

#include "huggingface_hub.h"
#include "llama_ros/llama.hpp"

/** @brief Real CPU inference tests with cancellation at known graph work. */
class PrecomputeCancelTest : public ::testing::Test {
protected:
  common_params params;
  int cache_reuse = 0;
  std::unique_ptr<llama_ros::Llama> engine;
  std::thread worker;
  std::atomic<int> prompt_microbatches{0};
  std::atomic<bool> cancel_prompt{false};
  uint64_t goal = 7101;
  int parallel = 1;
  int cache_ram_mib = 0;

  void SetUp() override {
    auto model = huggingface_hub::hf_hub_download_with_shards(
        "bartowski/SmolLM2-135M-Instruct-GGUF",
        "SmolLM2-135M-Instruct-Q6_K.gguf");
    ASSERT_TRUE(model.success);
    params.model.path = model.path;
    params.n_ctx = 512 * parallel;
    params.n_batch = 128;
    params.n_ubatch = 16;
    params.n_parallel = parallel;
    params.n_predict = 8;
    params.n_gpu_layers = 0;
    params.no_op_offload = true;
    params.cpuparams.n_threads = 1;
    params.cpuparams_batch.n_threads = 1;
    params.cache_prompt = true;
    params.n_cache_reuse = cache_reuse;
    params.cache_ram_mib = cache_ram_mib; // Zero proves direct KV reuse.
    params.warmup = false;
    params.cb_eval = [](ggml_tensor *tensor, bool ask, void *data) {
      auto &test = *static_cast<PrecomputeCancelTest *>(data);
      const bool first_layer = std::strcmp(tensor->name, "attn_norm-0") == 0;
      if (ask) {
        return first_layer;
      }
      if (first_layer && test.cancel_prompt.load() &&
          ++test.prompt_microbatches == 2) {
        // First microbatch is complete; the second is inside graph evaluation.
        test.engine->cancel_goal(test.goal);
      }
      return true;
    };
    params.cb_eval_user_data = this;
    engine = std::make_unique<llama_ros::Llama>(params, "");
    ASSERT_EQ(engine->supports_precompute(), parallel == 1);
    worker = std::thread([this] { engine->run_loop(); });
  }

  void TearDown() override {
    if (engine) {
      engine->cancel();
    }
    if (worker.joinable()) {
      worker.join();
    }
  }

  llama_ros::ServerSlot *reserve() {
    auto *slot = engine->wait_for_available_slot();
    slot->goal_id = goal;
    slot->state = llama_ros::SLOT_STATE_RESERVED;
    return slot;
  }

  std::string long_prompt() {
    std::string prompt;
    for (int i = 0; i < 60; ++i) {
      prompt += "The blue sky is clear. ";
    }
    return prompt;
  }
};

/** @brief Evaluation emits nothing, including when the prior goal generated. */
TEST_F(PrecomputeCancelTest, ZeroOutputAndIdenticalOrChangedTailReuse) {
  const std::string prompt = "The capital of France is Paris. The sky is blue.";
  reserve();
  int feedback = 0;
  auto evaluated = engine->generate_response(
      goal, prompt, params.sampling,
      [&feedback](llama_ros::CompletionOutput, llama_ros::ServerSlot *) {
        ++feedback;
      },
      {}, false, true);
  ASSERT_TRUE(evaluated.is_ok()) << evaluated.error();
  EXPECT_EQ(evaluated.value().n_decoded, 0);
  EXPECT_TRUE(evaluated.value().content.empty());
  EXPECT_TRUE(evaluated.value().tokens.empty());
  EXPECT_EQ(feedback, 0);
  auto *slot = reserve();
  const auto cached = slot->n_kv_cache;
  EXPECT_GT(cached, 1);
  auto same = engine->generate_response(goal, prompt, params.sampling, nullptr,
                                        {}, false, true);
  ASSERT_TRUE(same.is_ok()) << same.error();
  slot = reserve();
  EXPECT_EQ(slot->n_prompt_tokens_processed, 1);
  EXPECT_EQ(slot->n_kv_cache, cached);
  auto generated = engine->generate_response(
      goal, "The capital of France is Paris. The sky is cloudy.",
      params.sampling, nullptr, {}, false);
  ASSERT_TRUE(generated.is_ok()) << generated.error();
  EXPECT_GT(generated.value().n_decoded, 0);
  slot = reserve();
  EXPECT_LT(slot->n_prompt_tokens_processed, generated.value().n_prompt_tokens);
  auto after_generation = engine->generate_response(
      goal, prompt, params.sampling, nullptr, {}, false, true);
  ASSERT_TRUE(after_generation.is_ok()) << after_generation.error();
  EXPECT_TRUE(after_generation.value().tokens.empty());
  EXPECT_EQ(after_generation.value().n_decoded, 0);
}

/** @brief Abort inside prompt decode preserves earlier completed microbatches.
 */
TEST_F(PrecomputeCancelTest, CancelInsidePromptDecodePreservesCommittedPrefix) {
  const auto prompt = long_prompt();
  reserve();
  cancel_prompt = true;
  auto result = engine->generate_response(goal, prompt, params.sampling,
                                          nullptr, {}, false, true);
  ASSERT_TRUE(result.is_ok()) << result.error();
  ASSERT_GE(prompt_microbatches.load(), 2);
  EXPECT_EQ(result.value().stop, llama_ros::CANCEL);
  EXPECT_EQ(result.value().n_decoded, 0);
  EXPECT_TRUE(result.value().tokens.empty());
  auto *slot = reserve();
  const int committed = slot->n_kv_cache;
  EXPECT_GE(committed, params.n_ubatch);
  EXPECT_LT(committed, params.n_batch);
  EXPECT_EQ(slot->kv_cached_tokens.size(), static_cast<size_t>(committed));
  EXPECT_EQ(llama_memory_seq_pos_max(engine->get_memory(), slot->id) + 1,
            committed);
  cancel_prompt = false;
  auto next = engine->generate_response(goal, prompt, params.sampling, nullptr,
                                        {}, false, true);
  ASSERT_TRUE(next.is_ok()) << next.error();
  slot = reserve();
  EXPECT_EQ(slot->n_prompt_tokens_processed,
            next.value().n_prompt_tokens - committed);
}

/** @brief Cancellation survives registration and is cleared for the next goal.
 */
TEST_F(PrecomputeCancelTest, CancelBeforeRegistrationAndDuringGeneration) {
  reserve();
  engine->cancel_goal(goal);
  auto early = engine->generate_response(goal, "Hello world", params.sampling,
                                         nullptr, {}, false, true);
  ASSERT_TRUE(early.is_ok()) << early.error();
  EXPECT_EQ(early.value().stop, llama_ros::CANCEL);
  EXPECT_TRUE(early.value().tokens.empty());
  reserve();
  int feedback = 0;
  auto active = engine->generate_response(
      goal, "Count from one to ten:", params.sampling,
      [this, &feedback](llama_ros::CompletionOutput, llama_ros::ServerSlot *) {
        if (++feedback == 1) {
          engine->cancel_goal(goal);
        }
      },
      {}, false);
  ASSERT_TRUE(active.is_ok()) << active.error();
  EXPECT_EQ(active.value().stop, llama_ros::CANCEL);
  EXPECT_EQ(feedback, 1);
  reserve();
  auto next = engine->generate_response(goal, "The capital of France is",
                                        params.sampling, nullptr, {}, false);
  ASSERT_TRUE(next.is_ok()) << next.error();
  EXPECT_NE(next.value().stop, llama_ros::CANCEL);
  EXPECT_GT(next.value().n_decoded, 0);
}

/** @brief Shared decode batches must not be aborted for one canceled goal. */
class MultiSlotCancelTest : public PrecomputeCancelTest {
protected:
  void SetUp() override {
    parallel = 2;
    PrecomputeCancelTest::SetUp();
  }
};

TEST_F(MultiSlotCancelTest, CancelOneGoalKeepsOtherSlotUsable) {
  auto *first = reserve();
  const uint64_t first_goal = goal;
  ++goal;
  auto *second = reserve();
  ASSERT_NE(first->id, second->id);
  const uint64_t second_goal = goal;
  std::atomic<int> first_feedback{0};
  auto first_result = std::async(std::launch::async, [&] {
    return engine->generate_response(
        first_goal, "Count from one to ten:", params.sampling,
        [&](llama_ros::CompletionOutput, llama_ros::ServerSlot *) {
          if (++first_feedback == 1) {
            engine->cancel_goal(first_goal);
          }
        },
        {}, false);
  });
  auto second_result = std::async(std::launch::async, [&] {
    return engine->generate_response(second_goal, "The capital of France is",
                                     params.sampling, nullptr, {}, false);
  });
  auto canceled = first_result.get();
  auto completed = second_result.get();
  ASSERT_TRUE(canceled.is_ok()) << canceled.error();
  ASSERT_TRUE(completed.is_ok()) << completed.error();
  EXPECT_EQ(first_feedback.load(), 1);
  EXPECT_EQ(canceled.value().stop, llama_ros::CANCEL);
  EXPECT_NE(completed.value().stop, llama_ros::CANCEL);
  EXPECT_GT(completed.value().n_decoded, 0);
  ++goal;
  reserve();
  auto next = engine->generate_response(goal, "Hello world", params.sampling,
                                        nullptr, {}, false);
  ASSERT_TRUE(next.is_ok()) << next.error();
  EXPECT_GT(next.value().n_decoded, 0);
}

/** @brief Reused KV must produce the same greedy continuation as cold KV. */
TEST_F(PrecomputeCancelTest, ReusedChangedTailMatchesColdGeneration) {
  auto sampling = params.sampling;
  sampling.temp = 0.0f;
  sampling.seed = 42;
  const std::string prefix = "The capital of France is Paris. The capital of ";
  const std::string final_prompt = prefix + "Spain is";
  reserve();
  auto warm = engine->generate_response(goal, prefix + "Italy is Rome.",
                                        sampling, nullptr, {}, false, true);
  ASSERT_TRUE(warm.is_ok()) << warm.error();
  reserve();
  auto reused = engine->generate_response(goal, final_prompt, sampling, nullptr,
                                          {}, false);
  ASSERT_TRUE(reused.is_ok()) << reused.error();
  auto *idle = engine->wait_for_available_slot();
  EXPECT_LT(idle->n_prompt_tokens_processed, reused.value().n_prompt_tokens);

  // A separate freshly initialized context proves equality independently of
  // prefix bookkeeping and of reset()'s cache invalidation behavior.
  auto cold_params = params;
  cold_params.cb_eval = nullptr;
  cold_params.cb_eval_user_data = nullptr;
  llama_ros::Llama cold(cold_params, "");
  auto *slot = cold.wait_for_available_slot();
  slot->goal_id = goal;
  slot->state = llama_ros::SLOT_STATE_RESERVED;
  std::thread cold_worker([&] { cold.run_loop(); });
  auto fresh =
      cold.generate_response(goal, final_prompt, sampling, nullptr, {}, false);
  cold.cancel();
  cold_worker.join();
  ASSERT_TRUE(fresh.is_ok()) << fresh.error();
  EXPECT_EQ(reused.value().tokens, fresh.value().tokens);
  EXPECT_EQ(reused.value().content, fresh.value().content);
}

/** @brief Explicit context resets must also discard host-RAM snapshots. */
class ResetCacheTest : public PrecomputeCancelTest {
protected:
  void SetUp() override {
    cache_ram_mib = 64;
    PrecomputeCancelTest::SetUp();
  }
};

TEST_F(ResetCacheTest, ExplicitResetDiscardsAllCachedState) {
  const std::string prompt = "The capital of France is Paris. The sky is blue.";
  reserve();
  auto warm = engine->generate_response(goal, prompt, params.sampling, nullptr,
                                        {}, false, true);
  ASSERT_TRUE(warm.is_ok()) << warm.error();
  auto *slot = engine->wait_for_available_slot();
  ASSERT_GT(slot->n_kv_cache, 0);
  engine->reset();
  EXPECT_EQ(slot->n_kv_cache, 0);
  EXPECT_TRUE(slot->kv_cached_tokens.empty());
  EXPECT_TRUE(slot->checkpoints.empty());
  reserve();
  auto after = engine->generate_response(goal, prompt, params.sampling, nullptr,
                                         {}, false, true);
  ASSERT_TRUE(after.is_ok()) << after.error();
  slot = engine->wait_for_available_slot();
  // A surviving RAM snapshot would reduce this count despite empty slot KV.
  EXPECT_EQ(slot->n_prompt_tokens_processed, after.value().n_prompt_tokens);
}

/** @brief Invalidated token metadata cannot be reused as materialized KV. */
class InvalidatedCacheTest : public PrecomputeCancelTest {
protected:
  void SetUp() override {
    cache_reuse = 4;
    PrecomputeCancelTest::SetUp();
  }
};

TEST_F(InvalidatedCacheTest, ChunkReuseDoesNotResurrectInvalidKV) {
  const std::string prompt = "The capital of France is Paris. The sky is blue.";
  reserve();
  auto prepared = engine->generate_response(goal, prompt, params.sampling,
                                            nullptr, {}, false, true);
  ASSERT_TRUE(prepared.is_ok()) << prepared.error();
  auto *slot = engine->wait_for_available_slot();
  ASSERT_FALSE(slot->kv_cached_tokens.empty());
  // Invalidation preserves token labels for checkpoint matching, not valid KV.
  slot->invalidate_kv_cache();
  llama_memory_seq_rm(engine->get_memory(), slot->id, -1, -1);
  reserve();
  auto next = engine->generate_response(goal, prompt, params.sampling, nullptr,
                                        {}, false, true);
  ASSERT_TRUE(next.is_ok()) << next.error();
  slot = engine->wait_for_available_slot();
  EXPECT_EQ(slot->n_prompt_tokens_processed, next.value().n_prompt_tokens);
  EXPECT_EQ(llama_memory_seq_pos_max(engine->get_memory(), slot->id) + 1,
            next.value().n_prompt_tokens);
}
