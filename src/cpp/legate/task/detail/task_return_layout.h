/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/task/detail/return_value.h>

#include <cstdint>
#include <vector>

namespace legate::detail {

class TaskReturnLayoutForUnpack {
  static constexpr std::size_t BASE = 0x100;

 public:
  explicit TaskReturnLayoutForUnpack(std::size_t starting_offset = 0);

  [[nodiscard]] std::size_t total_size() const;

  [[nodiscard]] std::size_t next(std::size_t element_size, std::size_t alignment);

 protected:
  std::size_t current_offset_{};
};

class TaskReturnLayoutForPack : private TaskReturnLayoutForUnpack {
 public:
  using const_iterator = typename std::vector<std::size_t>::const_iterator;

  TaskReturnLayoutForPack() = default;
  explicit TaskReturnLayoutForPack(const std::vector<ReturnValue>& return_values);

  using TaskReturnLayoutForUnpack::total_size;

  [[nodiscard]] const_iterator begin() const;
  [[nodiscard]] const_iterator end() const;

 private:
  std::vector<std::size_t> offsets_{};
};

}  // namespace legate::detail

#include <legate/task/detail/task_return_layout.inl>
