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

#include "core/mapping/detail/default_mapper.h"

namespace legate::mapping::detail {

// Default mapper doesn't use the machine query interface
inline void DefaultMapper::set_machine(const MachineQueryInterface* /*machine*/) {}

inline TaskTarget DefaultMapper::task_target(const mapping::Task&,
                                             const std::vector<TaskTarget>& options)
{
  return options.front();
}

inline std::vector<mapping::StoreMapping> DefaultMapper::store_mappings(
  const mapping::Task& /*task*/, const std::vector<StoreTarget>& /*options*/)
{
  return {};
}

inline Scalar DefaultMapper::tunable_value(TunableID /*tunable_id*/)
{
  LEGATE_ABORT("Should not be called!");
  return Scalar{0};
}

}  // namespace legate::mapping::detail
