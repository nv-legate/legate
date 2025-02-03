/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/operation/detail/launcher_arg.h>

#include <legate/data/detail/array_kind.h>
#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/req_analyzer.h>
#include <legate/operation/detail/task_launcher.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/detail/buffer_builder.h>

namespace legate::detail {

void ScalarArg::pack(BufferBuilder& buffer) const { scalar_->pack(buffer); }

void RegionFieldArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  store_->pack(buffer);

  auto region   = store_->get_region_field()->region();
  auto field_id = store_->get_region_field()->field_id();

  buffer.pack<GlobalRedopID>(store_proj_->redop);
  buffer.pack<std::int32_t>(region.get_dim());
  buffer.pack<std::uint32_t>(analyzer.get_index(region, privilege_, *store_proj_, field_id));
  buffer.pack<std::uint32_t>(field_id);
}

void RegionFieldArg::analyze(StoreAnalyzer& analyzer)
{
  analyzer.insert(store_->get_region_field(), privilege_, *store_proj_);
}

std::optional<Legion::ProjectionID> RegionFieldArg::get_key_proj_id() const
{
  return store_proj_->is_key ? std::make_optional(store_proj_->proj_id) : std::nullopt;
}

void RegionFieldArg::perform_invalidations() const
{
  store_->get_region_field()->perform_invalidation_callbacks();
}

void OutputRegionArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  store_->pack(buffer);

  buffer.pack<GlobalRedopID>(GlobalRedopID{-1});
  buffer.pack<std::uint32_t>(store_->dim());
  // Need to cache the requirement index for post-processing
  requirement_index_ = analyzer.get_index(field_space_, field_id_);
  buffer.pack<std::uint32_t>(requirement_index_);
  buffer.pack<std::uint32_t>(field_id_);
}

void OutputRegionArg::analyze(StoreAnalyzer& analyzer)
{
  analyzer.insert(store_->dim(), field_space_, field_id_);
}

void OutputRegionArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  args.push_back(this);
}

void ScalarStoreArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  store_->pack(buffer);

  buffer.pack<GlobalRedopID>(redop_);
  buffer.pack<bool>(read_only_);
  buffer.pack<std::int32_t>(analyzer.get_index(future_));
  buffer.pack<std::uint32_t>(store_->type()->size());
  buffer.pack<std::uint32_t>(store_->type()->alignment());
  buffer.pack<std::uint64_t>(scalar_offset_);
  buffer.pack<std::uint64_t>(store_->get_storage()->extents().data());
}

void ScalarStoreArg::analyze(StoreAnalyzer& analyzer) { analyzer.insert(future_); }

void ReplicatedScalarStoreArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  store_->pack(buffer);

  buffer.pack<GlobalRedopID>(GlobalRedopID{-1});
  buffer.pack<bool>(read_only_);
  buffer.pack<std::int32_t>(analyzer.get_index(future_map_));
  buffer.pack<std::uint32_t>(store_->type()->size());
  buffer.pack<std::uint32_t>(store_->type()->alignment());
  buffer.pack<std::uint64_t>(scalar_offset_);
  buffer.pack<std::uint64_t>(store_->get_storage()->extents().data());
}

void ReplicatedScalarStoreArg::analyze(StoreAnalyzer& analyzer) { analyzer.insert(future_map_); }

void WriteOnlyScalarStoreArg::pack(BufferBuilder& buffer, const StoreAnalyzer& /*analyzer*/) const
{
  store_->pack(buffer);

  // redop
  buffer.pack<GlobalRedopID>(redop_);
  // read-only
  buffer.pack<bool>(false);
  // future index
  buffer.pack<std::int32_t>(-1);
  buffer.pack<std::uint32_t>(store_->type()->size());
  buffer.pack<std::uint32_t>(store_->type()->alignment());
  // field offset
  buffer.pack<std::uint64_t>(0);
  // TODO(wonchanl): the extents of an unbound scalar store are derived from the launch domain, but
  // this logic hasn't been implemented yet, as unbound scalar stores are not exposed to the API.
  // The code below works for the only use case in the runtime (approximate image computation)
  if (store_->unbound()) {
    buffer.pack<std::uint64_t>(std::vector<std::uint64_t>{1});
  } else {
    buffer.pack<std::uint64_t>(store_->get_storage()->extents().data());
  }
}

void BaseArrayArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  buffer.pack(to_underlying(ArrayKind::BASE));
  data_->pack(buffer, analyzer);

  const bool nullable = null_mask_ != nullptr;
  buffer.pack<bool>(nullable);
  if (nullable) {
    null_mask_->pack(buffer, analyzer);
  }
}

void BaseArrayArg::analyze(StoreAnalyzer& analyzer)
{
  data_->analyze(analyzer);
  if (null_mask_) {
    null_mask_->analyze(analyzer);
  }
}

std::optional<Legion::ProjectionID> BaseArrayArg::get_key_proj_id() const
{
  return data_->get_key_proj_id();
}

void BaseArrayArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  data_->record_unbound_stores(args);
  if (null_mask_ != nullptr) {
    null_mask_->record_unbound_stores(args);
  }
}

void BaseArrayArg::perform_invalidations() const
{
  // We don't need to invalidate any cached state for null masks
  data_->perform_invalidations();
}

void ListArrayArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  buffer.pack(to_underlying(ArrayKind::LIST));
  type_->pack(buffer);
  descriptor_->pack(buffer, analyzer);
  vardata_->pack(buffer, analyzer);
}

void ListArrayArg::analyze(StoreAnalyzer& analyzer)
{
  descriptor_->analyze(analyzer);
  vardata_->analyze(analyzer);
}

std::optional<Legion::ProjectionID> ListArrayArg::get_key_proj_id() const
{
  return vardata_->get_key_proj_id();
}

void ListArrayArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  descriptor_->record_unbound_stores(args);
  vardata_->record_unbound_stores(args);
}

void ListArrayArg::perform_invalidations() const
{
  descriptor_->perform_invalidations();
  vardata_->perform_invalidations();
}

void StructArrayArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  buffer.pack(to_underlying(ArrayKind::STRUCT));
  type_->pack(buffer);

  const bool nullable = null_mask_ != nullptr;
  buffer.pack<bool>(nullable);
  if (nullable) {
    null_mask_->pack(buffer, analyzer);
  }

  for (auto&& field : fields_) {
    field->pack(buffer, analyzer);
  }
}

void StructArrayArg::analyze(StoreAnalyzer& analyzer)
{
  if (null_mask_) {
    null_mask_->analyze(analyzer);
  }
  for (auto&& field : fields_) {
    field->analyze(analyzer);
  }
}

std::optional<Legion::ProjectionID> StructArrayArg::get_key_proj_id() const
{
  for (auto&& field : fields_) {
    auto proj_id = field->get_key_proj_id();
    if (proj_id.has_value()) {
      return proj_id;
    }
  }
  return std::nullopt;
}

void StructArrayArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  if (null_mask_) {
    null_mask_->record_unbound_stores(args);
  }
  for (auto&& field : fields_) {
    field->record_unbound_stores(args);
  }
}

void StructArrayArg::perform_invalidations() const
{
  if (null_mask_) {
    null_mask_->perform_invalidations();
  }
  for (auto&& field : fields_) {
    field->perform_invalidations();
  }
}

}  // namespace legate::detail
