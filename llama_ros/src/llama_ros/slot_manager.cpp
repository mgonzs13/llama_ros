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

#include "llama_ros/slot_manager.hpp"
#include "llama_ros/llama.hpp"
#include "llama_utils/logs.hpp"

namespace llama_ros {

SlotManager::SlotManager(std::vector<ServerSlot> &slots)
    : server_slots_(slots) {
      LLAMA_LOG_INFO("Slot Manager initialized with %d slots", slots.size());
    }

ServerSlot *SlotManager::get_available_slot() {
  for (auto &slot : server_slots_) {
    if (!slot.is_processing()) {
      return &slot;
    }
  }
  return nullptr;
}

ServerSlot *SlotManager::wait_for_available_slot() {
  std::unique_lock<std::mutex> lock(slot_mutex_);

  slot_cv_.wait(lock, [this] {
    for (auto &slot : server_slots_) {
      if (!slot.is_processing()) {
        return true;
      }
    }
    return false;
  });

  for (auto &slot : server_slots_) {
    if (!slot.is_processing()) {
      return &slot;
    }
  }

  return nullptr;
}

ServerSlot *SlotManager::get_slot_by_id(int id) {
  for (auto &slot : server_slots_) {
    if (slot.id == id) {
      return &slot;
    }
  }
  return nullptr;
}

ServerSlot *SlotManager::get_slot_by_gid(uint64_t gid) {
  for (auto &slot : server_slots_) {
    if (slot.goal_id == gid) {
      return &slot;
    }
  }
  return nullptr;
}

void SlotManager::release_slot(ServerSlot *slot) {
  {
    std::lock_guard<std::mutex> lock(slot_mutex_);
    slot->release();
  }
  slot_cv_.notify_one();
}

void SlotManager::notify_slot_available() {
  slot_cv_.notify_one();
}

} // namespace llama_ros
