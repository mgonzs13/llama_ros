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

#ifndef LLAMA_ROS__PROMPT_CACHE_HPP
#define LLAMA_ROS__PROMPT_CACHE_HPP

#include <cstddef>
#include <cstdint>
#include <list>
#include <vector>

#include "common.h"

namespace llama_ros {

/**
 * @brief One cached prompt state, keyed by its token sequence.
 */
struct PromptCacheEntry {
  std::vector<llama_token> tokens;
  common_prompt_checkpoint state;

  size_t size_bytes() const { return this->state.size(); }
};

/**
 * @brief Byte-bounded host-RAM cache of prompt/sequence states, mirroring
 * llama.cpp's server prompt cache.
 */
class PromptCache {
public:
  /**
   * @param limit_mib Cache size limit in MiB (< 0 = unlimited, 0 = disabled).
   * @param limit_tokens Token limit (0 = unlimited).
   */
  explicit PromptCache(int32_t limit_mib, size_t limit_tokens = 0);

  bool enabled() const { return this->limit_bytes_ != 0; }

  size_t size_bytes() const;
  size_t n_tokens() const;

  /**
   * @brief Reserve a new cache entry for @p tokens, evicting obsolete and
   * oldest entries as needed. Returns nullptr when the cache is disabled, the
   * prompt is empty, or it is already contained in another entry.
   * The caller fills the returned entry's state and calls update().
   */
  PromptCacheEntry *alloc(const std::vector<llama_token> &tokens);

  /**
   * @brief Find the cached entry with the best overlap for @p tokens
   * (upstream f_keep/f_sim scoring, 0.25 minimum keep ratio).
   */
  const PromptCacheEntry *
  find_best(const std::vector<llama_token> &tokens) const;

  void erase(const PromptCacheEntry *entry);

  /**
   * @brief Evict oldest entries until both limits are respected.
   */
  void update();

private:
  std::list<PromptCacheEntry> entries_; // front = oldest
  size_t limit_bytes_ = 0;              // 0 = disabled, SIZE_MAX = unlimited
  size_t limit_tokens_ = 0;             // 0 = unlimited
};

/**
 * @brief Common prefix length of two token sequences, stopping at
 * LLAMA_TOKEN_NULL.
 */
size_t common_prefix_len(const std::vector<llama_token> &a,
                         const std::vector<llama_token> &b);

} // namespace llama_ros

#endif // LLAMA_ROS__PROMPT_CACHE_HPP
