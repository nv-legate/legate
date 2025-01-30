/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
