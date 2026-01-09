/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/scalar.h>
#include <legate/mapping/mapping.h>
#include <legate/mapping/operation.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <optional>
#include <vector>

namespace legate::experimental::io::detail {

class Mapper final : public legate::mapping::Mapper {
 public:
  [[nodiscard]] std::vector<mapping::StoreMapping> store_mappings(
    const mapping::Task& task, const std::vector<mapping::StoreTarget>& options) override;
  [[nodiscard]] std::optional<std::size_t> allocation_pool_size(
    const mapping::Task& task, mapping::StoreTarget memory_kind) override;
  [[nodiscard]] legate::Scalar tunable_value(TunableID) override;
};

}  // namespace legate::experimental::io::detail
