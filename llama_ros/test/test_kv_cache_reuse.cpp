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
#include <vector>

#include "common.h"
#include "huggingface_hub.h"
#include "llama.h"
#include "llama_ros/server_slot.hpp"

using llama_ros::ServerSlot;

TEST(ServerSlotPrefixTest, CommonPrefixLenBasic) {
  ServerSlot slot;
  slot.kv_cached_tokens = {1, 2, 3, 4};

  EXPECT_EQ(slot.common_prefix_len({1, 2, 9}), 2u);
  EXPECT_EQ(slot.common_prefix_len({1, 2, 3, 4}), 4u);
  EXPECT_EQ(slot.common_prefix_len({9, 2, 3}), 0u);
}

TEST(ServerSlotPrefixTest, CommonPrefixLenStopsAtNullToken) {
  ServerSlot slot;
  slot.kv_cached_tokens = {1, 2, 3};

  EXPECT_EQ(slot.common_prefix_len({1, LLAMA_TOKEN_NULL, 3}), 1u);
}

TEST(ServerSlotPrefixTest, FindReusablePrefixRejectsMedia) {
  ServerSlot slot;
  slot.kv_cached_tokens = {1, 2, 3};
  slot.n_kv_cache = 3;
  slot.map_pos_to_media[0] = nullptr;

  EXPECT_EQ(slot.find_reusable_prefix({1, 2, 3}), 0u);
}

TEST(ServerSlotPrefixTest, FindReusablePrefixRejectsGeneratedTail) {
  ServerSlot slot;
  slot.kv_cached_tokens = {1, 2, 3};
  slot.n_kv_cache = 5; // previous request generated tokens

  EXPECT_EQ(slot.find_reusable_prefix({1, 2, 3, 4}), 0u);
}

TEST(ServerSlotPrefixTest, FindReusablePrefixFullCoverage) {
  ServerSlot slot;
  slot.kv_cached_tokens = {1, 2};
  slot.n_kv_cache = 2;

  EXPECT_EQ(slot.find_reusable_prefix({1, 2, 3}), 2u);
}

TEST(ServerSlotChunkReuseTest, ReusesMatchingChunkWithShift) {
  auto result = huggingface_hub::hf_hub_download_with_shards(
      "bartowski/SmolLM2-135M-Instruct-GGUF",
      "SmolLM2-135M-Instruct-Q6_K.gguf");
  ASSERT_TRUE(result.success) << "Failed to download model";
  ASSERT_FALSE(result.path.empty());

  common_params params;
  params.model.path = result.path;
  params.n_ctx = 256;
  params.n_batch = 128;
  params.n_ubatch = 128;
  params.cpuparams.n_threads = 1;
  params.cpuparams_batch.n_threads = 1;

  auto init = common_init_from_params(params);
  ASSERT_NE(init, nullptr);
  auto *ctx = init->context();
  ASSERT_NE(ctx, nullptr);
  auto mem = llama_get_memory(ctx);

  const std::vector<llama_token> cached = {10, 20, 30, 40, 50, 60, 70, 80};

  llama_batch batch = llama_batch_init((int32_t)cached.size(), 0, 1);
  for (size_t i = 0; i < cached.size(); i++) {
    common_batch_add(batch, cached[i], (llama_pos)i, {0},
                     i + 1 == cached.size());
  }
  ASSERT_EQ(llama_decode(ctx, batch), 0);
  llama_batch_free(batch);

  ServerSlot slot;
  slot.id = 0;
  slot.kv_cached_tokens = cached;

  // [20,30,40,50,60] is reused from cached[1..5]; the final 99 never matches
  const std::vector<llama_token> incoming = {20, 30, 40, 50, 60, 99};
  const size_t n_past = slot.reuse_kv_chunks(mem, incoming, 4);

  EXPECT_EQ(n_past, 5u);
  EXPECT_EQ(slot.kv_cached_tokens[0], 20);
  EXPECT_EQ(slot.kv_cached_tokens[4], 60);

  // tail beyond the reused prefix is dropped by the run_loop caller
  llama_memory_seq_rm(mem, slot.id, (llama_pos)n_past, -1);
}
