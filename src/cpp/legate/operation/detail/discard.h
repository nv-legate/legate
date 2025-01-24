/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <legate/operation/detail/operation.h>

#include <cstdint>

namespace legate::detail {

class Discard final : public Operation {
 public:
  Discard(std::uint64_t unique_id, Legion::LogicalRegion region, Legion::FieldID field_id);

  void launch() override;

  [[nodiscard]] Kind kind() const override;

 private:
  Legion::LogicalRegion region_{};
  Legion::FieldID field_id_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/discard.inl>
