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

#include "core/utilities/deserializer.h"

#include "core/data/detail/store.h"
#include "core/data/store.h"
#include "core/utilities/typedefs.h"

#include "legion/legion_c.h"
#include "legion/legion_c_util.h"

namespace legate {

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

std::vector<std::shared_ptr<detail::Array>> TaskDeserializer::unpack_arrays()
{
  std::vector<std::shared_ptr<detail::Array>> arrays;
  auto size = unpack<uint32_t>();

  arrays.reserve(size);
  for (uint32_t idx = 0; idx < size; ++idx) arrays.emplace_back(unpack_array());
  return arrays;
}

std::shared_ptr<detail::Array> TaskDeserializer::unpack_array()
{
  auto kind = static_cast<detail::ArrayKind>(unpack<int32_t>());

  switch (kind) {
    case detail::ArrayKind::BASE: return unpack_base_array();
    case detail::ArrayKind::LIST: return unpack_list_array();
    case detail::ArrayKind::STRUCT: return unpack_struct_array();
  }
  assert(false);
  return {};
}

std::shared_ptr<detail::BaseArray> TaskDeserializer::unpack_base_array()
{
  auto data      = unpack_store();
  auto nullable  = unpack<bool>();
  auto null_mask = nullable ? unpack_store() : nullptr;
  return std::make_shared<detail::BaseArray>(std::move(data), std::move(null_mask));
}

std::shared_ptr<detail::ListArray> TaskDeserializer::unpack_list_array()
{
  auto type = unpack_type();
  static_cast<void>(unpack<int32_t>());  // Unpack kind
  auto descriptor = unpack_base_array();
  auto vardata    = unpack_array();
  return std::make_shared<detail::ListArray>(
    std::move(type), std::move(descriptor), std::move(vardata));
}

std::shared_ptr<detail::StructArray> TaskDeserializer::unpack_struct_array()
{
  auto type = unpack_type();
  if (LegateDefined(LEGATE_USE_DEBUG)) assert(type->code == Type::Code::STRUCT);

  std::vector<std::shared_ptr<detail::Array>> fields;
  const auto& st_type = type->as_struct_type();
  auto nullable       = unpack<bool>();
  auto null_mask      = nullable ? unpack_store() : nullptr;

  fields.reserve(st_type.num_fields());
  for (uint32_t idx = 0; idx < st_type.num_fields(); ++idx) fields.emplace_back(unpack_array());
  return std::make_shared<detail::StructArray>(
    std::move(type), std::move(null_mask), std::move(fields));
}

std::shared_ptr<detail::Store> TaskDeserializer::unpack_store()
{
  auto is_future        = unpack<bool>();
  auto is_output_region = unpack<bool>();
  auto dim              = unpack<int32_t>();
  auto type             = unpack_type();
  auto transform        = unpack_transform();
  auto redop_id         = unpack<int32_t>();

  if (is_future) {
    auto fut = unpack<detail::FutureWrapper>();

    if (redop_id != -1 && !fut.valid()) fut.initialize_with_identity(redop_id);
    return std::make_shared<detail::Store>(
      dim, std::move(type), redop_id, std::move(fut), std::move(transform));
  }
  if (!is_output_region) {
    auto rf = unpack<detail::RegionField>();

    return std::make_shared<detail::Store>(
      dim, std::move(type), redop_id, std::move(rf), std::move(transform));
  }
  assert(redop_id == -1);
  auto out = unpack<detail::UnboundRegionField>();

  return std::make_shared<detail::Store>(
    dim, std::move(type), std::move(out), std::move(transform));
}

void TaskDeserializer::_unpack(detail::FutureWrapper& value)
{
  auto read_only    = unpack<bool>();
  auto future_index = unpack<int32_t>();
  auto field_size   = unpack<uint32_t>();
  auto domain       = unpack<Domain>();

  auto has_storage      = future_index >= 0;
  Legion::Future future = has_storage ? futures_[future_index] : Legion::Future{};
  value = detail::FutureWrapper{read_only, field_size, domain, std::move(future), has_storage};
}

void TaskDeserializer::_unpack(detail::RegionField& value)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = detail::RegionField{dim, regions_[idx], static_cast<Legion::FieldID>(fid)};
}

void TaskDeserializer::_unpack(detail::UnboundRegionField& value)
{
  static_cast<void>(unpack<int32_t>());  // dim
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  value = detail::UnboundRegionField{outputs_[idx], static_cast<Legion::FieldID>(fid)};
}

void TaskDeserializer::_unpack(comm::Communicator& value)
{
  auto future = futures_[0];
  futures_    = futures_.subspan(1);
  value       = comm::Communicator{future};
}

void TaskDeserializer::_unpack(Legion::PhaseBarrier& barrier)
{
  auto future   = futures_[0];
  futures_      = futures_.subspan(1);
  auto barrier_ = future.get_result<legion_phase_barrier_t>();
  barrier       = Legion::CObjectWrapper::unwrap(barrier_);
}

}  // namespace legate

namespace legate::mapping {

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

std::vector<std::shared_ptr<detail::Array>> TaskDeserializer::unpack_arrays()
{
  std::vector<std::shared_ptr<detail::Array>> arrays;
  auto size = unpack<uint32_t>();

  arrays.reserve(size);
  for (uint32_t idx = 0; idx < size; ++idx) arrays.emplace_back(unpack_array());
  return arrays;
}

std::shared_ptr<detail::Array> TaskDeserializer::unpack_array()
{
  auto kind = static_cast<legate::detail::ArrayKind>(unpack<int32_t>());

  switch (kind) {
    case legate::detail::ArrayKind::BASE: return unpack_base_array();
    case legate::detail::ArrayKind::LIST: return unpack_list_array();
    case legate::detail::ArrayKind::STRUCT: return unpack_struct_array();
  }
  assert(false);
  return {};
}

std::shared_ptr<detail::BaseArray> TaskDeserializer::unpack_base_array()
{
  auto data      = unpack_store();
  auto nullable  = unpack<bool>();
  auto null_mask = nullable ? unpack_store() : nullptr;

  return std::make_shared<detail::BaseArray>(std::move(data), std::move(null_mask));
}

std::shared_ptr<detail::ListArray> TaskDeserializer::unpack_list_array()
{
  auto type = unpack_type();
  static_cast<void>(unpack<int32_t>());  // Unpack kind
  auto descriptor = unpack_base_array();
  auto vardata    = unpack_array();
  return std::make_shared<detail::ListArray>(
    std::move(type), std::move(descriptor), std::move(vardata));
}

std::shared_ptr<detail::StructArray> TaskDeserializer::unpack_struct_array()
{
  auto type = unpack_type();
  if (LegateDefined(LEGATE_USE_DEBUG)) assert(type->code == Type::Code::STRUCT);

  std::vector<std::shared_ptr<detail::Array>> fields;
  const auto& st_type = type->as_struct_type();
  auto nullable       = unpack<bool>();
  auto null_mask      = nullable ? unpack_store() : nullptr;

  fields.reserve(st_type.num_fields());
  for (uint32_t idx = 0; idx < st_type.num_fields(); ++idx) fields.emplace_back(unpack_array());
  return std::make_shared<detail::StructArray>(
    std::move(type), std::move(null_mask), std::move(fields));
}

std::shared_ptr<detail::Store> TaskDeserializer::unpack_store()
{
  auto is_future        = unpack<bool>();
  auto is_output_region = unpack<bool>();
  auto dim              = unpack<int32_t>();
  auto type             = unpack_type();

  auto transform = unpack_transform();

  if (is_future) {
    // We still need to parse the reduction op
    static_cast<void>(unpack<int32_t>());
    auto fut = unpack<detail::FutureWrapper>();

    return std::make_shared<detail::Store>(
      dim, std::move(type), std::move(fut), std::move(transform));
  }
  auto redop_id = unpack<int32_t>();
  detail::RegionField rf;
  _unpack(rf, is_output_region);
  return std::make_shared<detail::Store>(
    runtime_, context_, dim, std::move(type), redop_id, rf, is_output_region, std::move(transform));
}

void TaskDeserializer::_unpack(detail::FutureWrapper& value)
{
  // We still need to deserialize these fields to get to the domain
  static_cast<void>(unpack<bool>());
  auto future_index = unpack<int32_t>();
  static_cast<void>(unpack<uint32_t>());
  auto domain = unpack<Domain>();
  value       = detail::FutureWrapper{static_cast<uint32_t>(future_index), domain};
}

void TaskDeserializer::_unpack(detail::RegionField& value, bool is_output_region)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();

  auto req = is_output_region ? &task_->output_regions[idx] : &task_->regions[idx];
  value    = detail::RegionField{req, dim, idx, static_cast<Legion::FieldID>(fid)};
}

CopyDeserializer::CopyDeserializer(const Legion::Copy* copy,
                                   std::vector<ReqsRef>&& all_requirements,
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
  if (LegateDefined(LEGATE_USE_DEBUG)) assert(curr_reqs_ != all_reqs_.end());
  req_index_offset_ += curr_reqs_->get().size();
  ++curr_reqs_;
}

void CopyDeserializer::_unpack(detail::Store& store)
{
  auto is_future        = unpack<bool>();
  auto is_output_region = unpack<bool>();
  auto dim              = unpack<int32_t>();
  auto type             = unpack_type();

  auto transform = unpack_transform();

  if (LegateDefined(LEGATE_USE_DEBUG)) assert(!is_future && !is_output_region);
  auto redop_id = unpack<int32_t>();
  detail::RegionField rf;

  _unpack(rf);
  store = detail::Store(
    runtime_, context_, dim, std::move(type), redop_id, rf, is_output_region, std::move(transform));
}

void CopyDeserializer::_unpack(detail::RegionField& value)
{
  auto dim = unpack<int32_t>();
  auto idx = unpack<uint32_t>();
  auto fid = unpack<int32_t>();
  auto req = &curr_reqs_->get()[idx];

  value = detail::RegionField{req, dim, idx + req_index_offset_, static_cast<Legion::FieldID>(fid)};
}

}  // namespace legate::mapping
