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

#include "core/task/detail/return_value.h"
#include "core/utilities/typedefs.h"

#include <vector>

namespace legate::detail {

class ReturnValues {
 public:
  ReturnValues() = default;
  explicit ReturnValues(std::vector<ReturnValue>&& return_values);

  [[nodiscard]] const ReturnValue& operator[](std::int32_t idx) const;

  [[nodiscard]] std::size_t legion_buffer_size() const;
  void legion_serialize(void* buffer) const;
  void legion_deserialize(const void* buffer);

  [[nodiscard]] static ReturnValue extract(const Legion::Future& future, std::uint32_t to_extract);

  // Calls the Legion postamble with an instance that packs all return values
  void finalize(Legion::Context legion_context) const;

 private:
  std::size_t buffer_size_{};
  std::vector<ReturnValue> return_values_{};
};

}  // namespace legate::detail

#include "core/task/detail/return.inl"
