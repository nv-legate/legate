/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/deserializer.h>

#include <legate/data/detail/array_kind.h>
#include <legate/data/detail/future_wrapper.h>
#include <legate/data/detail/physical_arrays/list_physical_array.h>
#include <legate/data/detail/physical_arrays/struct_physical_array.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/physical_stores/future_physical_store.h>
#include <legate/data/detail/physical_stores/region_physical_store.h>
#include <legate/data/detail/physical_stores/unbound_physical_store.h>
#include <legate/data/detail/physical_stores/unbound_region_field.h>
#include <legate/data/physical_store.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/typedefs.h>

#include <legion/bindings/c_bindings.h>
#include <legion/bindings/c_bindings_util.h>

#include <fmt/format.h>

namespace legate::detail {

TaskDeserializer::TaskDeserializer(const Legion::Task& task,
                                   const std::vector<Legion::PhysicalRegion>& regions)
  : BaseDeserializer{task.args, task.arglen},
    legion_task_{task},
    futures_{task.futures},
    regions_{regions}
{
  auto runtime = Legion::Runtime::get_runtime();
  auto ctx     = Legion::Runtime::get_context();
  runtime->get_output_regions(ctx, outputs_);
}

SmallVector<InternalSharedPtr<PhysicalArray>> TaskDeserializer::unpack_arrays()
{
  SmallVector<InternalSharedPtr<PhysicalArray>> arrays;
  const auto size = unpack<std::uint32_t>();

  arrays.reserve(size);
  for (std::uint32_t idx = 0; idx < size; ++idx) {
    arrays.emplace_back(unpack_array());
  }
  return arrays;
}

InternalSharedPtr<PhysicalArray> TaskDeserializer::unpack_array()
{
  switch (unpack<ArrayKind>()) {
    case ArrayKind::BASE: return unpack_base_array();
    case ArrayKind::LIST: return unpack_list_array();
    case ArrayKind::STRUCT: return unpack_struct_array();
  }
  return {};
}

InternalSharedPtr<BasePhysicalArray> TaskDeserializer::unpack_base_array()
{
  auto data      = unpack_store();
  auto nullable  = unpack<bool>();
  auto null_mask = nullable ? std::make_optional(unpack_store()) : std::nullopt;
  return make_internal_shared<BasePhysicalArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<ListPhysicalArray> TaskDeserializer::unpack_list_array()
{
  auto type = unpack_type_();
  static_cast<void>(unpack<ArrayKind>());  // Unpack kind
  auto descriptor = unpack_base_array();
  auto vardata    = unpack_array();
  return make_internal_shared<ListPhysicalArray>(
    std::move(type), std::move(descriptor), std::move(vardata));
}

InternalSharedPtr<StructPhysicalArray> TaskDeserializer::unpack_struct_array()
{
  auto type = unpack_type_();
  LEGATE_CHECK(type->code == Type::Code::STRUCT);

  SmallVector<InternalSharedPtr<PhysicalArray>> fields;
  const auto& st_type = dynamic_cast<const detail::StructType&>(*type);
  auto nullable       = unpack<bool>();
  auto null_mask      = nullable ? std::make_optional(unpack_store()) : std::nullopt;

  fields.reserve(st_type.num_fields());
  for (std::uint32_t idx = 0; idx < st_type.num_fields(); ++idx) {
    fields.emplace_back(unpack_array());
  }
  return make_internal_shared<StructPhysicalArray>(
    std::move(type), std::move(null_mask), std::move(fields));
}

InternalSharedPtr<PhysicalStore> TaskDeserializer::unpack_store()
{
  auto is_future = unpack<bool>();
  auto unbound   = unpack<bool>();
  auto dim       = unpack<std::int32_t>();
  auto type      = unpack_type_();
  auto transform = unpack_transform_();
  auto redop_id  = unpack<GlobalRedopID>();

  if (is_future) {
    auto fut = unpack<FutureWrapper>();

    if (redop_id != GlobalRedopID{-1} && !fut.valid()) {
      fut.initialize_with_identity(redop_id);
    }
    return make_internal_shared<FuturePhysicalStore>(
      dim, std::move(type), redop_id, std::move(fut), std::move(transform));
  }
  if (!unbound) {
    auto rf = unpack<RegionField>();

    return make_internal_shared<RegionPhysicalStore>(
      dim, std::move(type), redop_id, std::move(rf), std::move(transform));
  }
  LEGATE_CHECK(redop_id == GlobalRedopID{-1});
  auto out = unpack<UnboundRegionField>();

  return make_internal_shared<UnboundPhysicalStore>(
    dim, std::move(type), std::move(out), std::move(transform));
}

void TaskDeserializer::unpack_impl(FutureWrapper& value)
{
  const auto read_only       = unpack<bool>();
  const auto future_index    = unpack<std::int32_t>();
  const auto field_size      = unpack<std::uint32_t>();
  const auto field_alignment = unpack<std::uint32_t>();
  const auto field_offset    = unpack<std::uint64_t>();
  const auto domain          = unpack<Domain>();

  const auto has_storage = future_index >= 0;
  Legion::Future future  = has_storage ? futures_[future_index] : Legion::Future{};
  value =
    FutureWrapper{read_only, field_size, field_alignment, field_offset, domain, std::move(future)};
}

void TaskDeserializer::unpack_impl(RegionField& value)
{
  auto dim = unpack<std::int32_t>();
  auto idx = unpack<std::uint32_t>();
  auto fid = unpack<std::int32_t>();

  value =
    RegionField{dim,
                regions_[idx],
                static_cast<Legion::FieldID>(fid),
                legion_task_.get().regions[idx].partition != Legion::LogicalPartition::NO_PART};
}

void TaskDeserializer::unpack_impl(UnboundRegionField& value)
{
  static_cast<void>(unpack<std::int32_t>());  // dim
  auto idx = unpack<std::uint32_t>();
  auto fid = unpack<std::int32_t>();

  value = UnboundRegionField{
    outputs_[idx],
    static_cast<Legion::FieldID>(fid),
    legion_task_.get().output_regions[idx].partition != Legion::LogicalPartition::NO_PART};
}

void TaskDeserializer::unpack_impl(legate::comm::Communicator& value)
{
  auto future = futures_[0];
  futures_    = futures_.subspan(1);
  value       = legate::comm::Communicator{std::move(future)};
}

void TaskDeserializer::unpack_impl(Legion::PhaseBarrier& barrier)
{
  auto future        = futures_[0];
  futures_           = futures_.subspan(1);
  auto phase_barrier = future.get_result<legion_phase_barrier_t>();
  barrier            = Legion::CObjectWrapper::unwrap(std::move(phase_barrier));
}

}  // namespace legate::detail

namespace legate::mapping::detail {

MapperDataDeserializer::MapperDataDeserializer(const Legion::Mappable& mappable)
  : BaseDeserializer{mappable.mapper_data, mappable.mapper_data_size}
{
}

TaskDeserializer::TaskDeserializer(const Legion::Task& task,
                                   Legion::Mapping::MapperRuntime& runtime,
                                   Legion::Mapping::MapperContext context)
  : BaseDeserializer{task.args, task.arglen}, task_{task}, runtime_{runtime}, context_{context}
{
}

legate::detail::SmallVector<InternalSharedPtr<Array>> TaskDeserializer::unpack_arrays()
{
  legate::detail::SmallVector<InternalSharedPtr<Array>> arrays;
  const auto size = unpack<std::uint32_t>();

  arrays.reserve(size);
  for (std::uint32_t idx = 0; idx < size; ++idx) {
    arrays.emplace_back(unpack_array());
  }
  return arrays;
}

InternalSharedPtr<Array> TaskDeserializer::unpack_array()
{
  switch (unpack<legate::detail::ArrayKind>()) {
    case legate::detail::ArrayKind::BASE: return unpack_base_array();
    case legate::detail::ArrayKind::LIST: return unpack_list_array();
    case legate::detail::ArrayKind::STRUCT: return unpack_struct_array();
  }
  return {};
}

InternalSharedPtr<BaseArray> TaskDeserializer::unpack_base_array()
{
  auto data      = unpack_store();
  auto nullable  = unpack<bool>();
  auto null_mask = nullable ? std::make_optional(unpack_store()) : std::nullopt;

  return make_internal_shared<BaseArray>(std::move(data), std::move(null_mask));
}

InternalSharedPtr<ListArray> TaskDeserializer::unpack_list_array()
{
  auto type = unpack_type_();
  static_cast<void>(unpack<legate::detail::ArrayKind>());  // Unpack kind
  auto descriptor = unpack_base_array();
  auto vardata    = unpack_array();
  return make_internal_shared<ListArray>(
    std::move(type), std::move(descriptor), std::move(vardata));
}

InternalSharedPtr<StructArray> TaskDeserializer::unpack_struct_array()
{
  auto type = unpack_type_();
  LEGATE_CHECK(type->code == Type::Code::STRUCT);

  legate::detail::SmallVector<InternalSharedPtr<Array>> fields;
  const auto& st_type = dynamic_cast<const legate::detail::StructType&>(*type);
  auto nullable       = unpack<bool>();
  auto null_mask      = nullable ? std::make_optional(unpack_store()) : std::nullopt;

  fields.reserve(st_type.num_fields());
  for (std::uint32_t idx = 0; idx < st_type.num_fields(); ++idx) {
    fields.emplace_back(unpack_array());
  }
  return make_internal_shared<StructArray>(
    std::move(type), std::move(null_mask), std::move(fields));
}

InternalSharedPtr<Store> TaskDeserializer::unpack_store()
{
  auto is_future = unpack<bool>();
  auto unbound   = unpack<bool>();
  auto dim       = unpack<std::int32_t>();
  auto type      = unpack_type_();
  auto transform = unpack_transform_();

  if (is_future) {
    // We still need to parse the reduction op
    static_cast<void>(unpack<std::int32_t>());
    auto fut = unpack<FutureWrapper>();

    return make_internal_shared<Store>(dim, std::move(type), std::move(fut), std::move(transform));
  }
  auto redop_id = unpack<GlobalRedopID>();
  RegionField rf;
  unpack_impl(rf, unbound);
  return make_internal_shared<Store>(
    runtime_, context_, dim, std::move(type), redop_id, rf, unbound, std::move(transform));
}

void TaskDeserializer::unpack_impl(FutureWrapper& value)
{
  // We still need to deserialize these fields to get to the domain
  static_cast<void>(unpack<bool>());
  auto future_index = unpack<std::int32_t>();
  static_cast<void>(unpack<std::uint32_t>());
  static_cast<void>(unpack<std::uint32_t>());
  static_cast<void>(unpack<std::uint64_t>());
  auto domain = unpack<Domain>();
  value       = FutureWrapper{static_cast<std::uint32_t>(future_index), domain};
}

void TaskDeserializer::unpack_impl(RegionField& value, bool unbound)
{
  auto dim   = unpack<std::int32_t>();
  auto idx   = unpack<std::uint32_t>();
  auto fid   = unpack<std::int32_t>();
  auto&& req = unbound ? task_.get().output_regions[idx] : task_.get().regions[idx];

  value = RegionField{req, dim, idx, static_cast<Legion::FieldID>(fid), unbound};
}

CopyDeserializer::CopyDeserializer(const Legion::Copy& copy,
                                   Span<const ReqsRef> all_requirements,
                                   Legion::Mapping::MapperRuntime& runtime,
                                   Legion::Mapping::MapperContext context)
  : BaseDeserializer{copy.mapper_data, copy.mapper_data_size},
    all_reqs_{std::move(all_requirements)},
    curr_reqs_{all_reqs_.data()},
    runtime_{runtime},
    context_{context}
{
}

void CopyDeserializer::next_requirement_list()
{
  LEGATE_CHECK(curr_reqs_ != all_reqs_.data() + all_reqs_.size());
  req_index_offset_ += curr_reqs_->get().size();
  ++curr_reqs_;
}

void CopyDeserializer::unpack_impl(Store& store)
{
  auto is_future = unpack<bool>();
  auto unbound   = unpack<bool>();
  auto dim       = unpack<std::int32_t>();
  auto type      = unpack_type_();

  auto transform = unpack_transform_();

  LEGATE_CHECK(!is_future && !unbound);
  static_cast<void>(is_future);

  auto redop_id = unpack<GlobalRedopID>();
  auto rf       = unpack<RegionField>();

  store = Store{
    runtime_.get(), context_, dim, std::move(type), redop_id, rf, unbound, std::move(transform)};
}

void CopyDeserializer::unpack_impl(RegionField& value)
{
  auto dim   = unpack<std::int32_t>();
  auto idx   = unpack<std::uint32_t>();
  auto fid   = unpack<std::int32_t>();
  auto&& req = curr_reqs_->get()[idx];

  value = RegionField{
    req, dim, idx + req_index_offset_, static_cast<Legion::FieldID>(fid), false /*unbound*/};
}

}  // namespace legate::mapping::detail
