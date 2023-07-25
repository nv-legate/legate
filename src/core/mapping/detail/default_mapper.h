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

#include "core/mapping/mapping.h"

#pragma once

namespace legate::mapping::detail {

class DefaultMapper : public Mapper {
 public:
  virtual ~DefaultMapper() {}

 public:
  void set_machine(const MachineQueryInterface* machine) override;
  TaskTarget task_target(const mapping::Task& task,
                         const std::vector<TaskTarget>& options) override;
  std::vector<mapping::StoreMapping> store_mappings(
    const mapping::Task& task, const std::vector<StoreTarget>& options) override;
  Scalar tunable_value(TunableID tunable_id) override;
};

}  // namespace legate::mapping::detail
