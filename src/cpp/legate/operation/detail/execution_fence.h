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

#include "legate/operation/detail/operation.h"

#include <cstdint>

namespace legate::detail {

class ExecutionFence final : public Operation {
 public:
  ExecutionFence(std::uint64_t unique_id, bool block);

  void launch() override;

  [[nodiscard]] Kind kind() const override;

 private:
  bool block_{};
};

}  // namespace legate::detail

#include "legate/operation/detail/execution_fence.inl"
