/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/comm/pthread_barrier.h"

#include <atomic>
#include <cstdint>
#include <memory>

namespace legate::comm::coll {

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

}  // namespace legate::comm::coll

#include "core/comm/thread_comm.inl"
