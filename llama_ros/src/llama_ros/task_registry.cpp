// MIT License
//
// Copyright (c) 2023 Miguel Ángel González Santamarta
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

#include "llama_ros/task_registry.hpp"
#include "llama_ros/llama.hpp"
#include "llama_utils/logs.hpp"

namespace llama_ros {

std::future<ServerTaskResultPtr> TaskRegistry::register_pending(uint64_t goal_id) {
  std::lock_guard<std::mutex> lock(pending_mutex_);
  pending_.erase(goal_id);
  std::promise<ServerTaskResultPtr> promise;
  auto future = promise.get_future();
  pending_.emplace(goal_id, std::move(promise));
  return future;
}

void TaskRegistry::fulfill_pending(uint64_t goal_id, ServerTaskResultPtr result) {
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_.find(goal_id);
    if (it != pending_.end()) {
      it->second.set_value(std::move(result));
      pending_.erase(it);
    } else {
      LLAMA_LOG_WARN("Attempted to fulfill unknown goal_id: %lu", goal_id);
    }
  }
  mark_done(goal_id);
}

void TaskRegistry::fail_pending(uint64_t goal_id, const std::string &error) {
  {
    std::lock_guard<std::mutex> lock(pending_mutex_);
    auto it = pending_.find(goal_id);
    if (it != pending_.end()) {
      it->second.set_exception(std::make_exception_ptr(std::runtime_error(error)));
      pending_.erase(it);
    } else {
      LLAMA_LOG_WARN("Attempted to fail unknown goal_id: %lu", goal_id);
    }
  }
  mark_done(goal_id);
}

void TaskRegistry::mark_done(uint64_t goal_id) {
  {
    std::lock_guard<std::mutex> lock(done_mutex_);
    done_queue_.push(goal_id);
  }
  done_cv_.notify_one();
}

uint64_t TaskRegistry::wait_for_done() {
  std::unique_lock<std::mutex> lock(done_mutex_);
  done_cv_.wait(lock, [this] { return !done_queue_.empty(); });
  
  uint64_t goal_id = done_queue_.front();
  done_queue_.pop();
  return goal_id;
}

bool TaskRegistry::has_done_tasks() {
  std::lock_guard<std::mutex> lock(done_mutex_);
  return !done_queue_.empty();
}

} // namespace llama_ros
