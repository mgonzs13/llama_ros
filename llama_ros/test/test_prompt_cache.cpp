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

#include "llama_ros/prompt_cache.hpp"

using llama_ros::PromptCache;
using llama_ros::PromptCacheEntry;

TEST(PromptCacheTest, DisabledAtZeroMib) {
  PromptCache cache(0);

  EXPECT_FALSE(cache.enabled());
  EXPECT_EQ(cache.alloc({1, 2, 3}), nullptr);
  EXPECT_EQ(cache.find_best({1, 2, 3}), nullptr);
}

TEST(PromptCacheTest, AllocAndEvictByBytes) {
  PromptCache cache(1); // 1 MiB

  PromptCacheEntry *e1 = cache.alloc({1, 2, 3});
  ASSERT_NE(e1, nullptr);
  e1->state.data_tgt.resize(700 * 1024);
  cache.update();
  EXPECT_EQ(cache.size_bytes(), 700u * 1024);

  PromptCacheEntry *e2 = cache.alloc({4, 5, 6});
  ASSERT_NE(e2, nullptr);
  e2->state.data_tgt.resize(700 * 1024);
  cache.update();

  // e1 was evicted to stay within the byte limit
  EXPECT_EQ(cache.size_bytes(), 700u * 1024);
  EXPECT_EQ(cache.find_best({1, 2, 3}), nullptr);
  ASSERT_NE(cache.find_best({4, 5, 6}), nullptr);
}

TEST(PromptCacheTest, FindsBestEntryAndRejectsLowOverlap) {
  PromptCache cache(-1); // unlimited

  PromptCacheEntry *e1 = cache.alloc({1, 2, 3, 4, 5, 6, 7, 8});
  ASSERT_NE(e1, nullptr);
  e1->state.data_tgt.resize(10);

  const PromptCacheEntry *best = cache.find_best({1, 2, 3, 4, 9});
  ASSERT_NE(best, nullptr);
  EXPECT_EQ(best, e1);

  // lcp = 1 / 8 tokens -> f_keep = 0.125 < 0.25 threshold
  EXPECT_EQ(cache.find_best({1, 9, 9, 9, 9}), nullptr);
}

TEST(PromptCacheTest, SkipsContainedPrompt) {
  PromptCache cache(-1);

  PromptCacheEntry *e1 = cache.alloc({1, 2, 3, 4, 5});
  ASSERT_NE(e1, nullptr);
  e1->state.data_tgt.resize(10);

  // new prompt fully contained in an existing entry -> skip
  EXPECT_EQ(cache.alloc({1, 2, 3}), nullptr);
}

TEST(PromptCacheTest, DropsObsoleteEntries) {
  PromptCache cache(-1);

  PromptCacheEntry *e1 = cache.alloc({1, 2, 3});
  ASSERT_NE(e1, nullptr);
  e1->state.data_tgt.resize(10);

  // new prompt extends e1 -> e1 is obsolete and removed
  PromptCacheEntry *e2 = cache.alloc({1, 2, 3, 4, 5});
  ASSERT_NE(e2, nullptr);
  e2->state.data_tgt.resize(10);

  EXPECT_EQ(cache.n_tokens(), 5u); // only e2 remains
  const PromptCacheEntry *best = cache.find_best({1, 2, 3, 4, 5});
  ASSERT_NE(best, nullptr);
  EXPECT_EQ(best, e2);
}
