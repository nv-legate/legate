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

#include "core/mapping/detail/default_mapper.h"

namespace legate::mapping::detail {

// Default mapper doesn't use the machine query interface
void DefaultMapper::set_machine(const MachineQueryInterface* machine) {}

TaskTarget DefaultMapper::task_target(const mapping::Task& task,
                                      const std::vector<TaskTarget>& options)
{
  return options.front();
}

std::vector<mapping::StoreMapping> DefaultMapper::store_mappings(
  const mapping::Task& task, const std::vector<StoreTarget>& options)
{
  return {};
}

Scalar DefaultMapper::tunable_value(TunableID tunable_id)
{
  LEGATE_ABORT;
  return Scalar(0);
}

}  // namespace legate::mapping::detail
