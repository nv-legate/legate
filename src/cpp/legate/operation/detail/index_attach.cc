/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/index_attach.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/zip.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>

namespace legate::detail {

IndexAttach::IndexAttach(std::uint64_t unique_id,
                         InternalSharedPtr<LogicalRegionField> region_field,
                         std::uint32_t dim,
                         std::vector<Legion::LogicalRegion> subregions,
                         std::vector<InternalSharedPtr<ExternalAllocation>> allocations,
                         InternalSharedPtr<mapping::detail::DimOrdering> ordering)
  : Operation{unique_id},
    region_field_{std::move(region_field)},
    dim_{dim},
    subregions_{std::move(subregions)},
    allocations_{std::move(allocations)},
    ordering_{std::move(ordering)}
{
  region_field_->mark_attached();
}

void IndexAttach::launch()
{
  auto* runtime = Runtime::get_runtime();
  auto launcher = Legion::IndexAttachLauncher{
    LEGION_EXTERNAL_INSTANCE, region_field_->region(), false /*restricted*/};

  for (auto&& [subregion, allocation] : zip_equal(subregions_, allocations_)) {
    launcher.add_external_resource(subregion, allocation->resource());
  }

  static_assert(std::is_same_v<decltype(launcher.provenance), std::string>,
                "Don't use to_string() below");
  launcher.provenance                               = provenance().to_string();
  launcher.constraints.ordering_constraint.ordering = ordering_->generate_legion_dims(dim_);
  launcher.constraints.ordering_constraint.ordering.push_back(DIM_F);
  launcher.constraints.field_constraint.field_set =
    std::vector<Legion::FieldID>{region_field_->field_id()};
  launcher.constraints.field_constraint.contiguous = false;
  launcher.constraints.field_constraint.inorder    = false;
  launcher.privilege_fields.insert(region_field_->field_id());

  auto external_resources = runtime->get_legion_runtime()->attach_external_resources(
    runtime->get_legion_context(), launcher);
  region_field_->attach(std::move(external_resources), std::move(allocations_));
}

}  // namespace legate::detail
