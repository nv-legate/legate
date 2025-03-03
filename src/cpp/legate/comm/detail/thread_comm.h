/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/detail/pthread_barrier.h>

#include <atomic>
#include <cstdint>
#include <memory>

namespace legate::detail::comm::coll {

class ThreadComm {
 public:
  using atomic_buffer_type = std::atomic<const void*>;
  using atomic_displ_type  = std::atomic<const int*>;

  void init(std::int32_t global_comm_size);
  void finalize(std::int32_t global_comm_size, bool is_finalizer);
  void clear() noexcept;
  void barrier_local();

  [[nodiscard]] bool ready() const;
  [[nodiscard]] const atomic_buffer_type* buffers() const;
  [[nodiscard]] atomic_buffer_type* buffers();
  [[nodiscard]] const atomic_displ_type* displs() const;
  [[nodiscard]] atomic_displ_type* displs();

 private:
  std::unique_ptr<atomic_buffer_type[]> buffers_{};
  std::unique_ptr<atomic_displ_type[]> displs_{};
  std::atomic<bool> ready_flag_{};
  std::atomic<std::int32_t> entered_finalize_{};
  pthread_barrier_t barrier_{};
};

}  // namespace legate::detail::comm::coll

#include <legate/comm/detail/thread_comm.inl>
