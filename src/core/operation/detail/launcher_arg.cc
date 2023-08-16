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

#include "core/operation/detail/launcher_arg.h"

#include "core/data/detail/array_kind.h"
#include "core/data/detail/logical_region_field.h"
#include "core/operation/detail/req_analyzer.h"
#include "core/operation/detail/task_launcher.h"
#include "core/type/detail/type_info.h"
#include "core/utilities/detail/buffer_builder.h"

namespace legate::detail {

void ScalarArg::pack(BufferBuilder& buffer) const { scalar_.pack(buffer); }

RegionFieldArg::RegionFieldArg(LogicalStore* store,
                               Legion::PrivilegeMode privilege,
                               std::unique_ptr<ProjectionInfo> proj_info)
  : store_(store), privilege_(privilege), proj_info_(std::move(proj_info))
{
}

void RegionFieldArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  store_->pack(buffer);

  auto region   = store_->get_region_field()->region();
  auto field_id = store_->get_region_field()->field_id();

  buffer.pack<int32_t>(proj_info_->redop);
  buffer.pack<int32_t>(region.get_dim());
  buffer.pack<uint32_t>(analyzer.get_index(region, privilege_, *proj_info_));
  buffer.pack<uint32_t>(field_id);
}

void RegionFieldArg::analyze(StoreAnalyzer& analyzer)
{
  analyzer.insert(store_->get_region_field(), privilege_, *proj_info_);
}

std::optional<Legion::ProjectionID> RegionFieldArg::get_key_proj_id() const
{
  return LEGATE_CORE_KEY_STORE_TAG == proj_info_->tag ? std::make_optional(proj_info_->proj_id)
                                                      : std::nullopt;
}

void RegionFieldArg::perform_invalidations() const
{
  store_->get_region_field()->perform_invalidation_callbacks();
}

OutputRegionArg::OutputRegionArg(LogicalStore* store, Legion::FieldSpace field_space)
  : store_(store), field_space_(field_space)
{
  // TODO: We should reuse field ids here
  field_id_ = Runtime::get_runtime()->allocate_field(field_space_, store->type()->size());
}

void OutputRegionArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  store_->pack(buffer);

  buffer.pack<int32_t>(-1);
  buffer.pack<int32_t>(store_->dim());
  // Need to cache the requirement index for post-processing
  requirement_index_ = analyzer.get_index(field_space_, field_id_);
  buffer.pack<uint32_t>(requirement_index_);
  buffer.pack<uint32_t>(field_id_);
}

void OutputRegionArg::analyze(StoreAnalyzer& analyzer)
{
  analyzer.insert(store_->dim(), field_space_, field_id_);
}

void OutputRegionArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  args.push_back(this);
}

FutureStoreArg::FutureStoreArg(LogicalStore* store,
                               bool read_only,
                               bool has_storage,
                               Legion::ReductionOpID redop)
  : store_(store), read_only_(read_only), has_storage_(has_storage), redop_(redop)
{
}

void FutureStoreArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  store_->pack(buffer);

  buffer.pack<int32_t>(redop_);
  buffer.pack<bool>(read_only_);
  buffer.pack<int32_t>(has_storage_ ? analyzer.get_index(store_->get_future()) : -1);
  buffer.pack<uint32_t>(store_->type()->size());
  buffer.pack<size_t>(store_->get_storage()->extents().data());
}

void FutureStoreArg::analyze(StoreAnalyzer& analyzer)
{
  if (!has_storage_) return;
  analyzer.insert(store_->get_future());
}

BaseArrayArg::BaseArrayArg(std::unique_ptr<Analyzable> data) : data_(std::move(data)) {}

BaseArrayArg::BaseArrayArg(std::unique_ptr<Analyzable> data, std::unique_ptr<Analyzable> null_mask)
  : data_(std::move(data)), null_mask_(std::move(null_mask))
{
}

void BaseArrayArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  buffer.pack<int32_t>(static_cast<int32_t>(ArrayKind::BASE));
  data_->pack(buffer, analyzer);

  bool nullable = null_mask_ != nullptr;
  buffer.pack<bool>(nullable);
  if (nullable) { null_mask_->pack(buffer, analyzer); }
}

void BaseArrayArg::analyze(StoreAnalyzer& analyzer)
{
  data_->analyze(analyzer);
  if (null_mask_ != nullptr) null_mask_->analyze(analyzer);
}

std::optional<Legion::ProjectionID> BaseArrayArg::get_key_proj_id() const
{
  return data_->get_key_proj_id();
}

void BaseArrayArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  data_->record_unbound_stores(args);
  if (null_mask_ != nullptr) null_mask_->record_unbound_stores(args);
}

void BaseArrayArg::perform_invalidations() const
{
  // We don't need to invalidate any cached state for null masks
  data_->perform_invalidations();
}

ListArrayArg::ListArrayArg(std::shared_ptr<Type> type,
                           std::unique_ptr<Analyzable> descriptor,
                           std::unique_ptr<Analyzable> vardata)
  : type_(std::move(type)), descriptor_(std::move(descriptor)), vardata_(std::move(vardata))
{
}

void ListArrayArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  buffer.pack<int32_t>(static_cast<int32_t>(ArrayKind::LIST));
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

StructArrayArg::StructArrayArg(std::shared_ptr<Type> type,
                               std::unique_ptr<Analyzable> null_mask,
                               std::vector<std::unique_ptr<Analyzable>>&& fields)
  : type_(std::move(type)), null_mask_(std::move(null_mask)), fields_(std::move(fields))
{
}

void StructArrayArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  buffer.pack<int32_t>(static_cast<int32_t>(ArrayKind::STRUCT));
  type_->pack(buffer);

  bool nullable = null_mask_ != nullptr;
  buffer.pack<bool>(nullable);
  if (nullable) { null_mask_->pack(buffer, analyzer); }

  for (auto& field : fields_) { field->pack(buffer, analyzer); }
}

void StructArrayArg::analyze(StoreAnalyzer& analyzer)
{
  if (null_mask_ != nullptr) { null_mask_->analyze(analyzer); }
  for (auto& field : fields_) { field->analyze(analyzer); }
}

std::optional<Legion::ProjectionID> StructArrayArg::get_key_proj_id() const
{
  for (auto& field : fields_) {
    auto proj_id = field->get_key_proj_id();
    if (proj_id.has_value()) { return proj_id; }
  }
  return std::nullopt;
}

void StructArrayArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  if (null_mask_ != nullptr) { null_mask_->record_unbound_stores(args); }
  for (auto& field : fields_) { field->record_unbound_stores(args); }
}

void StructArrayArg::perform_invalidations() const
{
  if (null_mask_ != nullptr) { null_mask_->perform_invalidations(); }
  for (auto& field : fields_) { field->perform_invalidations(); }
}

}  // namespace legate::detail
