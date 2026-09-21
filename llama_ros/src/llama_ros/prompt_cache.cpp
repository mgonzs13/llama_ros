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

#include "llama_ros/prompt_cache.hpp"

#include <algorithm>

namespace llama_ros {

size_t common_prefix_len(const std::vector<llama_token> &a,
                         const std::vector<llama_token> &b) {
  const size_t max_check = std::min(a.size(), b.size());
  size_t i = 0;
  while (i < max_check && a[i] == b[i] && b[i] != LLAMA_TOKEN_NULL) {
    i++;
  }
  return i;
}

PromptCache::PromptCache(int32_t limit_mib, size_t limit_tokens) {
  if (limit_mib < 0) {
    this->limit_bytes_ = SIZE_MAX;
  } else {
    this->limit_bytes_ = static_cast<size_t>(limit_mib) * 1024 * 1024;
  }
  this->limit_tokens_ = limit_tokens;
}

size_t PromptCache::size_bytes() const {
  size_t res = 0;
  for (const auto &entry : this->entries_) {
    res += entry.size_bytes();
  }
  return res;
}

size_t PromptCache::n_tokens() const {
  size_t res = 0;
  for (const auto &entry : this->entries_) {
    res += entry.tokens.size();
  }
  return res;
}

PromptCacheEntry *PromptCache::alloc(const std::vector<llama_token> &tokens) {
  if (!this->enabled() || tokens.empty()) {
    return nullptr;
  }

  // skip if the new prompt is already fully contained in an entry
  for (const auto &entry : this->entries_) {
    if (common_prefix_len(entry.tokens, tokens) == tokens.size()) {
      return nullptr;
    }
  }

  // drop obsolete entries fully contained in the new prompt
  for (auto it = this->entries_.begin(); it != this->entries_.end();) {
    if (common_prefix_len(it->tokens, tokens) == it->tokens.size()) {
      it = this->entries_.erase(it);
    } else {
      ++it;
    }
  }

  this->entries_.push_back(
      PromptCacheEntry{tokens, common_prompt_checkpoint{}});
  return &this->entries_.back();
}

const PromptCacheEntry *
PromptCache::find_best(const std::vector<llama_token> &tokens) const {
  if (!this->enabled() || tokens.empty()) {
    return nullptr;
  }

  const PromptCacheEntry *best = nullptr;
  float best_keep = -1.0f;
  float best_sim = -1.0f;

  for (const auto &entry : this->entries_) {
    if (entry.tokens.empty()) {
      continue;
    }

    const size_t lcp = common_prefix_len(entry.tokens, tokens);
    const float keep =
        static_cast<float>(lcp) / static_cast<float>(entry.tokens.size());
    const float sim =
        static_cast<float>(lcp) / static_cast<float>(tokens.size());

    // don't trash large prompts
    if (keep < 0.25f) {
      continue;
    }

    if (keep > best_keep && sim > best_sim) {
      best = &entry;
      best_keep = keep;
      best_sim = sim;
    }
  }

  return best;
}

void PromptCache::erase(const PromptCacheEntry *entry) {
  if (entry == nullptr) {
    return;
  }

  for (auto it = this->entries_.begin(); it != this->entries_.end(); ++it) {
    if (&*it == entry) {
      this->entries_.erase(it);
      return;
    }
  }
}

void PromptCache::update() {
  if (!this->enabled()) {
    return;
  }

  while (!this->entries_.empty() && this->size_bytes() > this->limit_bytes_) {
    this->entries_.pop_front();
  }

  if (this->limit_tokens_ > 0) {
    while (!this->entries_.empty() && this->n_tokens() > this->limit_tokens_) {
      this->entries_.pop_front();
    }
  }
}

} // namespace llama_ros
