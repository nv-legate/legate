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

#include <legate_defines.h>

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/operation.h>
#include <legate/utilities/macros.h>

#include <cstdint>

namespace legate::detail {

class Strategy;

class Timing final : public Operation {
 public:
  enum class Precision : std::uint8_t {
    MICRO,
    NANO,
  };

  Timing(std::uint64_t unique_id, Precision precision, InternalSharedPtr<LogicalStore> store);

  void launch() override;
  // This method never gets invoked, but we define it just to appease compilers that don't like
  // partially overridden overloads (e.g., NVCC)
  void launch(Strategy* strategy) override;

  [[nodiscard]] Kind kind() const override;

 private:
  Precision precision_{Precision::MICRO};
  InternalSharedPtr<LogicalStore> store_{};
};

}  // namespace legate::detail

#include <legate/operation/detail/timing.inl>
