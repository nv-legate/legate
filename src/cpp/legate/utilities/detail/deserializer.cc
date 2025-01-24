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

#include <legate/utilities/detail/deserializer.h>

#include <legate/data/detail/physical_store.h>
#include <legate/data/physical_store.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/typedefs.h>

#include <legion/legion_c.h>
#include <legion/legion_c_util.h>

#include <fmt/format.h>

#include <stdexcept>

namespace legate::detail {

std::pair<void*, std::size_t> align_for_unpack_impl(void* ptr,
                                                    std::size_t capacity,
                                                    std::size_t bytes,
                                                    std::size_t align)
{
  const auto orig_avail_space = std::min(bytes + align - 1, capacity);
  auto avail_space            = orig_avail_space;

  if (!std::align(align, bytes, ptr, avail_space)) {
    // If we get here, it means that someone did not pack the value correctly, likely without
    // first aligning the pointer!
    throw TracedException<std::runtime_error>{fmt::format(
      "Failed to align buffer {} (of size: {}) to {}-byte alignment (remaining capacity: {})",
      ptr,
      bytes,
      align,
      capacity)};
  }
  return {ptr, orig_avail_space - avail_space};
}

TaskDeserializer::TaskDeserializer(const Legion::Task* task,
                                   const std::vector<Legion::PhysicalRegion>& regions)
  : BaseDeserializer{task->args, task->arglen},
    futures_{task->futures.data(), task->futures.size()},
    regions_{regions.data(), regions.size()}
{
  auto runtime = Legion::Runtime::get_runtime();
  auto ctx     = Legion::Runtime::get_context();
  runtime->get_output_regions(ctx, outputs_);
}

std::vector<InternalSharedPtr<PhysicalArray>> TaskDeserializer::unpack_arrays()
{
  std::vector<InternalSharedPtr<PhysicalArray>> arrays;
  auto size = unpack<std::uint32_t>();

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
  auto null_mask = nullable ? unpack_store() : nullptr;
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

  std::vector<InternalSharedPtr<PhysicalArray>> fields;
  const auto& st_type = dynamic_cast<const detail::StructType&>(*type);
  auto nullable       = unpack<bool>();
  auto null_mask      = nullable ? unpack_store() : nullptr;

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
    return make_internal_shared<PhysicalStore>(
      dim, std::move(type), redop_id, std::move(fut), std::move(transform));
  }
  if (!unbound) {
    auto rf = unpack<RegionField>();

    return make_internal_shared<PhysicalStore>(
      dim, std::move(type), redop_id, std::move(rf), std::move(transform));
  }
  LEGATE_CHECK(redop_id == GlobalRedopID{-1});
  auto out = unpack<UnboundRegionField>();

  return make_internal_shared<PhysicalStore>(
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

  value = RegionField{dim, regions_[idx], static_cast<Legion::FieldID>(fid)};
}

void TaskDeserializer::unpack_impl(UnboundRegionField& value)
{
  static_cast<void>(unpack<std::int32_t>());  // dim
  auto idx = unpack<std::uint32_t>();
  auto fid = unpack<std::int32_t>();

  value = UnboundRegionField{outputs_[idx], static_cast<Legion::FieldID>(fid)};
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

MapperDataDeserializer::MapperDataDeserializer(const Legion::Mappable* mappable)
  : BaseDeserializer{mappable->mapper_data, mappable->mapper_data_size}
{
}

TaskDeserializer::TaskDeserializer(const Legion::Task* task,
                                   Legion::Mapping::MapperRuntime* runtime,
                                   Legion::Mapping::MapperContext context)
  : BaseDeserializer{task->args, task->arglen}, task_{task}, runtime_{runtime}, context_{context}
{
}

std::vector<InternalSharedPtr<Array>> TaskDeserializer::unpack_arrays()
{
  std::vector<InternalSharedPtr<Array>> arrays;
  auto size = unpack<std::uint32_t>();

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
  auto null_mask = nullable ? unpack_store() : nullptr;

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

  std::vector<InternalSharedPtr<Array>> fields;
  const auto& st_type = dynamic_cast<const legate::detail::StructType&>(*type);
  auto nullable       = unpack<bool>();
  auto null_mask      = nullable ? unpack_store() : nullptr;

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
  auto dim = unpack<std::int32_t>();
  auto idx = unpack<std::uint32_t>();
  auto fid = unpack<std::int32_t>();
  auto req = unbound ? &task_->output_regions[idx] : &task_->regions[idx];

  value = RegionField{req, dim, idx, static_cast<Legion::FieldID>(fid), unbound};
}

CopyDeserializer::CopyDeserializer(const Legion::Copy* copy,
                                   Span<const ReqsRef> all_requirements,
                                   Legion::Mapping::MapperRuntime* runtime,
                                   Legion::Mapping::MapperContext context)
  : BaseDeserializer{copy->mapper_data, copy->mapper_data_size},
    all_reqs_{std::move(all_requirements)},
    curr_reqs_{all_reqs_.begin()},
    runtime_{runtime},
    context_{context}
{
}

void CopyDeserializer::next_requirement_list()
{
  LEGATE_CHECK(curr_reqs_ != all_reqs_.end());
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
  RegionField rf;

  unpack_impl(rf);
  store =
    Store{runtime_, context_, dim, std::move(type), redop_id, rf, unbound, std::move(transform)};
}

void CopyDeserializer::unpack_impl(RegionField& value)
{
  auto dim = unpack<std::int32_t>();
  auto idx = unpack<std::uint32_t>();
  auto fid = unpack<std::int32_t>();
  auto req = &curr_reqs_->get()[idx];

  value = RegionField{
    req, dim, idx + req_index_offset_, static_cast<Legion::FieldID>(fid), false /*unbound*/};
}

}  // namespace legate::mapping::detail
