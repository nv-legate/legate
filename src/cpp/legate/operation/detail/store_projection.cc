/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/store_projection.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/core_ids.h>

namespace legate::detail {

template <>
Legion::RegionRequirement StoreProjection::create_requirement<true>(
  const Legion::LogicalRegion& region,
  const std::vector<Legion::FieldID>& fields,
  Legion::PrivilegeMode privilege,
  bool is_single) const
{
  auto requirement = [&] {
    auto parent = Runtime::get_runtime().find_parent_region(region);

    if (LEGION_REDUCE == privilege) {
      return Legion::RegionRequirement{
        region, static_cast<Legion::ReductionOpID>(redop), LEGION_EXCLUSIVE, std::move(parent)};
    }

    const auto coherence_property = (!is_single && (privilege & LEGION_WRITE_PRIV) != 0)
                                      ? LEGION_COLLECTIVE_EXCLUSIVE
                                      : LEGION_EXCLUSIVE;

    return Legion::RegionRequirement{region, privilege, coherence_property, std::move(parent)};
  }();
  requirement.add_fields(fields);
  return requirement;
}

template <>
Legion::RegionRequirement StoreProjection::create_requirement<false>(
  const Legion::LogicalRegion& region,
  const std::vector<Legion::FieldID>& fields,
  Legion::PrivilegeMode privilege,
  bool /*is_single*/) const
{
  if (Legion::LogicalPartition::NO_PART == partition) {
    return create_requirement<true>(region, fields, privilege, /* is_single */ false);
  }

  auto requirement = [&] {
    auto parent = Runtime::get_runtime().find_parent_region(region);
    if (LEGION_REDUCE == privilege) {
      return Legion::RegionRequirement{partition,
                                       proj_id,
                                       static_cast<Legion::ReductionOpID>(redop),
                                       LEGION_EXCLUSIVE,
                                       std::move(parent)};
    }
    return Legion::RegionRequirement{
      partition, proj_id, privilege, LEGION_EXCLUSIVE, std::move(parent)};
  }();
  requirement.add_fields(fields);
  return requirement;
}

}  // namespace legate::detail
