/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/attach.h>

#include <legate/runtime/detail/runtime.h>

namespace legate::detail {

Attach::Attach(std::uint64_t unique_id,
               InternalSharedPtr<LogicalRegionField> region_field,
               std::uint32_t dim,
               InternalSharedPtr<ExternalAllocation> allocation,
               InternalSharedPtr<mapping::detail::DimOrdering> ordering)
  : Operation{unique_id},
    region_field_{std::move(region_field)},
    dim_{dim},
    allocation_{std::move(allocation)},
    ordering_{std::move(ordering)}
{
}

void Attach::launch()
{
  auto&& runtime = Runtime::get_runtime();

  auto launcher       = Legion::AttachLauncher{LEGION_EXTERNAL_INSTANCE,
                                         region_field_->region(),
                                         region_field_->region(),
                                         false /*restricted*/,
                                         !allocation_->read_only() /*mapped*/};
  launcher.collective = true;  // each shard will attach a full local copy of the entire buffer
  static_assert(std::is_same_v<decltype(launcher.provenance), std::string>,
                "Don't use to_string() below");
  launcher.provenance                               = provenance().to_string();
  launcher.constraints.ordering_constraint.ordering = ordering_->generate_legion_dims(dim_);
  launcher.constraints.field_constraint.field_set =
    std::vector<Legion::FieldID>{region_field_->field_id()};
  launcher.constraints.field_constraint.contiguous = false;
  launcher.constraints.field_constraint.inorder    = false;
  launcher.privilege_fields.insert(region_field_->field_id());
  launcher.external_resource = allocation_->resource();

  auto pr =
    runtime.get_legion_runtime()->attach_external_resource(runtime.get_legion_context(), launcher);
  // no need to wait on the returned PhysicalRegion, since we're not inline-mapping
  // but we can keep it around and remap it later if the user asks
  region_field_->attach(std::move(pr), std::move(allocation_));
}

}  // namespace legate::detail
