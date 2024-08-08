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

#include "core/mapping/detail/core_mapper.h"

#include <cstdlib>
#include <vector>

namespace legate::mapping::detail {

// This is a custom mapper implementation that only has to map
// start-up tasks associated with the Legate core, no one else
// should be overriding this mapper so we burry it in here
class CoreMapper final : public Mapper {
 public:
  void set_machine(const legate::mapping::MachineQueryInterface* machine) override;
  [[nodiscard]] legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override;
  [[nodiscard]] std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override;
  [[nodiscard]] legate::Scalar tunable_value(legate::TunableID tunable_id) override;
};

void CoreMapper::set_machine(const legate::mapping::MachineQueryInterface* /*m*/) {}

TaskTarget CoreMapper::task_target(const legate::mapping::Task&,
                                   const std::vector<TaskTarget>& options)
{
  return options.front();
}

std::vector<legate::mapping::StoreMapping> CoreMapper::store_mappings(
  const legate::mapping::Task&, const std::vector<StoreTarget>&)
{
  return {};
}

Scalar CoreMapper::tunable_value(TunableID /*tunable_id*/)
{
  // Illegal tunable variable
  LEGATE_ABORT("Tunable values are no longer supported");
  return Scalar{0};
}

std::unique_ptr<Mapper> create_core_mapper() { return std::make_unique<CoreMapper>(); }

}  // namespace legate::mapping::detail
