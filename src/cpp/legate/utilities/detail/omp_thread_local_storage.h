/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace legate::detail {

// Simple STL vector-based thread local storage for OpenMP threads to avoid false sharing
template <typename VAL>
class OMPThreadLocalStorage {
 public:
  using value_type = VAL;

 private:
  static constexpr std::size_t CACHE_LINE_SIZE = 64;
  // Round the element size to the nearest multiple of cache line size
  static constexpr std::size_t PER_THREAD_SIZE =
    (sizeof(VAL) + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE * CACHE_LINE_SIZE;

 public:
  explicit OMPThreadLocalStorage(std::uint32_t num_threads);

  VAL& operator[](std::uint32_t idx);

 private:
  std::vector<std::int8_t> storage_{};
};

}  // namespace legate::detail

#include <legate/utilities/detail/omp_thread_local_storage.inl>
