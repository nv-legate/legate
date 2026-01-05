/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/return_value.h>
#include <legate/task/detail/task_return_layout.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <vector>

namespace legate::detail {

class TaskReturn {
 public:
  static constexpr std::size_t ALIGNMENT = 16;

  TaskReturn() = default;
  explicit TaskReturn(std::vector<ReturnValue>&& return_values);

  [[nodiscard]] std::size_t buffer_size() const;
  void pack(void* buffer) const;

  // Calls the Legion postamble with an instance that packs all return values
  void finalize(Legion::Context legion_context, bool skip_device_ctx_sync) const;

 private:
  std::vector<ReturnValue> return_values_{};
  TaskReturnLayoutForPack layout_{};
};

}  // namespace legate::detail

#include <legate/task/detail/task_return.inl>
