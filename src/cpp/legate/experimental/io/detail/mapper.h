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

#include <legate/data/scalar.h>
#include <legate/mapping/mapping.h>
#include <legate/mapping/operation.h>
#include <legate/utilities/typedefs.h>

#include <vector>

namespace legate::experimental::io::detail {

class Mapper final : public legate::mapping::Mapper {
 public:
  [[nodiscard]] mapping::TaskTarget task_target(
    const mapping::Task& task, const std::vector<mapping::TaskTarget>& options) override;
  [[nodiscard]] std::vector<mapping::StoreMapping> store_mappings(
    const mapping::Task& task, const std::vector<mapping::StoreTarget>& options) override;
  [[nodiscard]] legate::Scalar tunable_value(TunableID) override;
};

}  // namespace legate::experimental::io::detail
