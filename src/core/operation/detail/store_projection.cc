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

#include "core/operation/detail/store_projection.h"

#include "core/runtime/detail/runtime.h"

namespace legate::detail {

template <>
void BaseStoreProjection::populate_requirement<true>(Legion::RegionRequirement& requirement,
                                                     const Legion::LogicalRegion& region,
                                                     const std::vector<Legion::FieldID>& fields,
                                                     Legion::PrivilegeMode privilege,
                                                     bool is_key,
                                                     bool is_single) const
{
  auto parent = Runtime::get_runtime()->find_parent_region(region);
  auto tag    = static_cast<Legion::MappingTagID>(is_key ? LEGATE_CORE_KEY_STORE_TAG : 0);

  // REVIEW
  // must explicitly call the destructor here, otherwise this whole business is UB!!!
  // see https://eel.is/c++draft/basic.life#1
  requirement.~RegionRequirement();
  if (LEGION_REDUCE == privilege) {
    new (&requirement) Legion::RegionRequirement{region, redop, LEGION_EXCLUSIVE, parent, tag};
  } else if (!is_single && (privilege & LEGION_WRITE_PRIV) != 0) {
    new (&requirement)
      Legion::RegionRequirement{region, privilege, LEGION_COLLECTIVE_EXCLUSIVE, parent, tag};
  } else {
    new (&requirement) Legion::RegionRequirement{region, privilege, LEGION_EXCLUSIVE, parent, tag};
  }
  requirement.add_fields(fields);
}

template <>
void BaseStoreProjection::populate_requirement<false>(Legion::RegionRequirement& requirement,
                                                      const Legion::LogicalRegion& region,
                                                      const std::vector<Legion::FieldID>& fields,
                                                      Legion::PrivilegeMode privilege,
                                                      bool is_key,
                                                      bool /*is_single*/) const
{
  if (Legion::LogicalPartition::NO_PART == partition) {
    populate_requirement<true>(requirement, region, fields, privilege, is_key, false);
    return;
  }

  auto parent = Runtime::get_runtime()->find_parent_region(region);
  auto tag    = static_cast<Legion::MappingTagID>(is_key ? LEGATE_CORE_KEY_STORE_TAG : 0);

  // see above
  requirement.~RegionRequirement();
  if (LEGION_REDUCE == privilege) {
    new (&requirement)
      Legion::RegionRequirement{partition, proj_id, redop, LEGION_EXCLUSIVE, parent, tag};
  } else {
    new (&requirement)
      Legion::RegionRequirement{partition, proj_id, privilege, LEGION_EXCLUSIVE, parent, tag};
  }
  requirement.add_fields(fields);
}

}  // namespace legate::detail
