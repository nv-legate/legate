/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/launcher_arg.h>

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
  buffer.pack<std::int32_t>(static_cast<std::int32_t>(store_->get_storage()->dim()));
  buffer.pack<std::uint32_t>(analyzer.get_index(region, privilege_, store_proj_, field_id));
  buffer.pack<std::uint32_t>(field_id);
}

void RegionFieldArg::analyze(StoreAnalyzer& analyzer) const
{
  analyzer.insert(store_->get_region_field(), privilege_, store_proj_);
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
  analyzer.insert(store_->dim(), field_space_, field_id_, proj_id_, color_space_);
}

void OutputRegionArg::record_unbound_stores(SmallVector<const OutputRegionArg*>& args) const
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
  buffer.pack<std::uint64_t>(store_->get_storage()->extents());
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
  buffer.pack<std::uint64_t>(store_->get_storage()->extents());
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
  if (store_->deferred_bound()) {
    static constexpr std::uint64_t extents[1] = {1};

    buffer.pack<std::uint64_t>(extents);
  } else {
    buffer.pack<std::uint64_t>(store_->get_storage()->extents());
  }
}

}  // namespace legate::detail
