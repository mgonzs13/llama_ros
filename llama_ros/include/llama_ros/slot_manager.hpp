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

#ifndef LLAMA_ROS__SLOT_MANAGER_HPP
#define LLAMA_ROS__SLOT_MANAGER_HPP

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <vector>

namespace llama_ros {

// Forward declaration
class ServerSlot;

/**
 * @brief Manages the allocation and lifecycle of ServerSlot instances.
 *
 * This class encapsulates slot management logic, including allocation,
 * release, and lookup operations. It provides thread-safe access to slots.
 */
class SlotManager {
public:
  /**
   * @brief Constructor for SlotManager.
   *
   * @param slots Reference to the vector of ServerSlot objects to manage.
   */
  explicit SlotManager(std::vector<ServerSlot> &slots);

  /**
   * @brief Destructor for SlotManager.
   */
  ~SlotManager() = default;

  /**
   * @brief Attempts to get an available (idle) slot without blocking.
   *
   * @return Pointer to an available slot, or nullptr if none available.
   */
  ServerSlot *get_available_slot();

  /**
   * @brief Waits for an available slot to become available.
   *
   * This method blocks until a slot becomes available.
   *
   * @return Pointer to an available slot.
   */
  ServerSlot *wait_for_available_slot();

  /**
   * @brief Gets a slot by its ID.
   *
   * @param id The slot ID.
   * @return Pointer to the slot, or nullptr if not found.
   */
  ServerSlot *get_slot_by_id(int id);

  /**
   * @brief Gets a slot by its goal ID.
   *
   * @param gid The goal ID associated with the slot.
   * @return Pointer to the slot, or nullptr if not found.
   */
  ServerSlot *get_slot_by_gid(uint64_t gid);

  /**
   * @brief Releases a slot, making it available for reuse.
   *
   * @param slot Pointer to the slot to release.
   */
  void release_slot(ServerSlot *slot);

  /**
   * @brief Notifies waiting threads that a slot may be available.
   */
  void notify_slot_available();

private:
  std::vector<ServerSlot> &server_slots_;
  std::mutex slot_mutex_;
  std::condition_variable slot_cv_;
};

} // namespace llama_ros

#endif // LLAMA_ROS__SLOT_MANAGER_HPP
