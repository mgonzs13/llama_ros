// MIT License
//
// Copyright (c) 2026 Miguel Ángel González Santamarta
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

#ifndef LLAMA_ROS__TASK_REGISTRY_HPP
#define LLAMA_ROS__TASK_REGISTRY_HPP

#include <condition_variable>
#include <cstdint>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>

namespace llama_ros {

// Forward declaration
struct ServerTaskResult;
using ServerTaskResultPtr = std::unique_ptr<ServerTaskResult>;

/**
 * @brief Manages asynchronous task registration and completion.
 *
 * This class handles the promise/future pattern for async tasks,
 * allowing tasks to be registered with a goal ID and fulfilled later
 * when results are available.
 */
class TaskRegistry {
public:
  /**
   * @brief Constructor for TaskRegistry.
   */
  TaskRegistry() = default;

  /**
   * @brief Destructor for TaskRegistry.
   */
  ~TaskRegistry() = default;

  /**
   * @brief Registers a new pending task and returns a future for the result.
   *
   * @param goal_id The unique identifier for the task.
   * @return A future that will contain the task result.
   */
  std::future<ServerTaskResultPtr> register_pending(uint64_t goal_id);

  /**
   * @brief Fulfills a pending task with a result.
   *
   * @param goal_id The unique identifier for the task.
   * @param result The result to fulfill the task with.
   */
  void fulfill_pending(uint64_t goal_id, ServerTaskResultPtr result);

  /**
   * @brief Fails a pending task with an error message.
   *
   * @param goal_id The unique identifier for the task.
   * @param error The error message.
   */
  void fail_pending(uint64_t goal_id, const std::string &error);

  /**
   * @brief Marks a task as done and notifies waiting threads.
   *
   * @param goal_id The unique identifier for the task.
   */
  void mark_done(uint64_t goal_id);

  /**
   * @brief Waits for the next completed task and returns its goal ID.
   *
   * @return The goal ID of the next completed task.
   */
  uint64_t wait_for_done();

  /**
   * @brief Checks if there are any done tasks in the queue.
   *
   * @return True if there are done tasks, false otherwise.
   */
  bool has_done_tasks();

private:
  std::unordered_map<uint64_t, std::promise<ServerTaskResultPtr>> pending_;
  std::mutex pending_mutex_;

  std::mutex done_mutex_;
  std::condition_variable done_cv_;
  std::queue<uint64_t> done_queue_;
};

} // namespace llama_ros

#endif // LLAMA_ROS__TASK_REGISTRY_HPP
