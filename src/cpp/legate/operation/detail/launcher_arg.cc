/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/launcher_arg.h>

#include <legate/data/detail/array_kind.h>
#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/store_analyzer.h>
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

  buffer.pack<GlobalRedopID>(store_proj_.redop);
  buffer.pack<std::int32_t>(region.get_dim());
  buffer.pack<std::uint32_t>(analyzer.get_index(region, privilege_, store_proj_, field_id));
  buffer.pack<std::uint32_t>(field_id);
}

void RegionFieldArg::analyze(StoreAnalyzer& analyzer) const
{
  analyzer.insert(store_->get_region_field(), privilege_, store_proj_);
}

std::optional<Legion::ProjectionID> RegionFieldArg::get_key_proj_id() const
{
  return store_proj_.is_key ? std::make_optional(store_proj_.proj_id) : std::nullopt;
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

void OutputRegionArg::analyze(StoreAnalyzer& analyzer) const
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

void ScalarStoreArg::analyze(StoreAnalyzer& analyzer) const { analyzer.insert(future_); }

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

void ReplicatedScalarStoreArg::analyze(StoreAnalyzer& analyzer) const
{
  analyzer.insert(future_map_);
}

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
  std::visit([&](const auto& arg) { arg.pack(buffer, analyzer); }, data_);

  const bool nullable = null_mask_.has_value();
  buffer.pack<bool>(nullable);
  if (nullable) {
    std::visit([&](const auto& arg) { arg.pack(buffer, analyzer); }, *null_mask_);
  }
}

void BaseArrayArg::analyze(StoreAnalyzer& analyzer) const
{
  std::visit([&](auto& arg) { arg.analyze(analyzer); }, data_);
  if (null_mask_.has_value()) {
    std::visit([&](auto& arg) { arg.analyze(analyzer); }, *null_mask_);
  }
}

std::optional<Legion::ProjectionID> BaseArrayArg::get_key_proj_id() const
{
  return std::visit([&](const auto& arg) { return arg.get_key_proj_id(); }, data_);
}

void BaseArrayArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  std::visit([&](const auto& arg) { return arg.record_unbound_stores(args); }, data_);
  if (null_mask_.has_value()) {
    std::visit([&](const auto& arg) { return arg.record_unbound_stores(args); }, *null_mask_);
  }
}

void BaseArrayArg::perform_invalidations() const
{
  // We don't need to invalidate any cached state for null masks
  std::visit([&](const auto& arg) { return arg.perform_invalidations(); }, data_);
}

ListArrayArg::ListArrayArg(InternalSharedPtr<Type> type,
                           ArrayAnalyzable&& descriptor,
                           ArrayAnalyzable&& vardata)
  : type_{std::move(type)},
    pimpl_{std::make_unique<Impl>(std::move(descriptor), std::move(vardata))}
{
}

void ListArrayArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  buffer.pack(to_underlying(ArrayKind::LIST));
  type_->pack(buffer);
  std::visit([&](const auto& arg) { arg.pack(buffer, analyzer); }, pimpl_->descriptor);
  std::visit([&](const auto& arg) { arg.pack(buffer, analyzer); }, pimpl_->vardata);
}

void ListArrayArg::analyze(StoreAnalyzer& analyzer) const
{
  std::visit([&](auto& arg) { arg.analyze(analyzer); }, pimpl_->descriptor);
  std::visit([&](auto& arg) { arg.analyze(analyzer); }, pimpl_->vardata);
}

std::optional<Legion::ProjectionID> ListArrayArg::get_key_proj_id() const
{
  return std::visit([&](const auto& arg) { return arg.get_key_proj_id(); }, pimpl_->vardata);
}

void ListArrayArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  std::visit([&](const auto& arg) { return arg.record_unbound_stores(args); }, pimpl_->descriptor);
  std::visit([&](const auto& arg) { return arg.record_unbound_stores(args); }, pimpl_->vardata);
}

void ListArrayArg::perform_invalidations() const
{
  std::visit([&](const auto& arg) { return arg.perform_invalidations(); }, pimpl_->descriptor);
  std::visit([&](const auto& arg) { return arg.perform_invalidations(); }, pimpl_->vardata);
}

void StructArrayArg::pack(BufferBuilder& buffer, const StoreAnalyzer& analyzer) const
{
  buffer.pack(to_underlying(ArrayKind::STRUCT));
  type_->pack(buffer);

  const bool nullable = null_mask_.has_value();
  buffer.pack<bool>(nullable);
  if (nullable) {
    std::visit([&](const auto& arg) { arg.pack(buffer, analyzer); }, *null_mask_);
  }

  for (auto&& field : fields_) {
    std::visit([&](const auto& arg) { arg.pack(buffer, analyzer); }, field);
  }
}

void StructArrayArg::analyze(StoreAnalyzer& analyzer) const
{
  if (null_mask_.has_value()) {
    std::visit([&](auto& arg) { arg.analyze(analyzer); }, *null_mask_);
  }
  for (auto&& field : fields_) {
    std::visit([&](auto& arg) { arg.analyze(analyzer); }, field);
  }
}

std::optional<Legion::ProjectionID> StructArrayArg::get_key_proj_id() const
{
  for (auto&& field : fields_) {
    auto proj_id = std::visit([&](const auto& arg) { return arg.get_key_proj_id(); }, field);
    if (proj_id.has_value()) {
      return proj_id;
    }
  }
  return std::nullopt;
}

void StructArrayArg::record_unbound_stores(std::vector<const OutputRegionArg*>& args) const
{
  if (null_mask_.has_value()) {
    std::visit([&](const auto& arg) { return arg.record_unbound_stores(args); }, *null_mask_);
  }
  for (auto&& field : fields_) {
    std::visit([&](const auto& arg) { return arg.record_unbound_stores(args); }, field);
  }
}

void StructArrayArg::perform_invalidations() const
{
  if (null_mask_.has_value()) {
    std::visit([&](const auto& arg) { return arg.perform_invalidations(); }, *null_mask_);
  }
  for (auto&& field : fields_) {
    std::visit([&](const auto& arg) { return arg.perform_invalidations(); }, field);
  }
}

}  // namespace legate::detail
