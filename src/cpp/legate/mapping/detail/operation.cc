/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/operation.h>

#include <legate/mapping/detail/mapping.h>
#include <legate/runtime/detail/library.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/deserializer.h>

namespace legate::mapping::detail {

Mappable::Mappable(private_tag, MapperDataDeserializer dez)
  // TODO(jfaibussowit):
  // Streaming generation must come first. It is the only thing used in the hot-loop of
  // select_tasks_to_map(), and we don't want to spend a bunch of time deserializing other
  // stuff we won't use in that function.
  //
  // Another idea would be to implement a generic "deserialize only X, not Y or Z", but due to
  // alignment handling this is not quite so simple as jumping over sizeof(Y) + sizeof(Z)
  // bytes. It would also need to handle dynamic sizes, e.g. if we serialize a vector of
  // stuff.
  //
  // We would need to fake-unpack everything in order to properly handle all of this, or create
  // a better serialization format that includes all of this offset information in some kind of
  // header.
  //
  // So as a workaround, streaming generation must come first, because at least then, we don't
  // need to do any offset fiddling.
  : streaming_gen_{dez.unpack<std::optional<legate::detail::StreamingGeneration>>()},
    machine_{dez.unpack<Machine>()},
    sharding_id_{dez.unpack<std::uint32_t>()},
    priority_{dez.unpack<std::int32_t>()}
{
}

/* static */ std::optional<legate::detail::StreamingGeneration>
Mappable::deserialize_only_streaming_generation(const Legion::Mappable& mappable)
{
  return MapperDataDeserializer{mappable}
    .unpack<std::optional<legate::detail::StreamingGeneration>>();
}

Task::Task(const Legion::Task& task,
           Legion::Mapping::MapperRuntime& runtime,
           Legion::Mapping::MapperContext context)
  : Mappable{task}, task_{task}
{
  TaskDeserializer dez{task, runtime, context};
  library_             = dez.unpack<legate::detail::Library*>();
  task_info_           = dez.unpack<legate::detail::TaskInfo*>();
  inputs_              = dez.unpack_arrays();
  outputs_             = dez.unpack_arrays();
  reductions_          = dez.unpack_arrays();
  scalars_             = dez.unpack_scalars();
  future_size_         = dez.unpack<std::size_t>();
  can_raise_exception_ = dez.unpack<bool>();

  if (legion_task().tag ==
      static_cast<Legion::MappingTagID>(legate::detail::CoreMappingTag::TREE_REDUCE)) {
    inputs_.erase(
      std::remove_if(inputs_.begin(), inputs_.end(), [](const auto& inp) { return !inp->valid(); }),
      inputs_.end());
  }
}

LocalTaskID Task::task_id() const
{
  return library().get_local_task_id(static_cast<GlobalTaskID>(legion_task().task_id));
}

Legion::VariantID Task::legion_task_variant() const
{
  return legate::detail::to_underlying(to_variant_code(target()));
}

// ==========================================================================================

Copy::Copy(const Legion::Copy& copy,
           Legion::Mapping::MapperRuntime& runtime,
           Legion::Mapping::MapperContext context)
  : copy_{copy}
{
  const auto reqs = {std::cref(copy.src_requirements),
                     std::cref(copy.dst_requirements),
                     std::cref(copy.src_indirect_requirements),
                     std::cref(copy.dst_indirect_requirements)};
  CopyDeserializer dez{copy, reqs, runtime, context};

  // Mappable
  //
  // Cannot use Mappable ctor here because both the Mappable and Copy data live in the same
  // serdez buffer. So when Mappable() unpacks its stuff CopyDeserializer doesn't know that the
  // buffer should be advanced. And we cannot advance the pointer ourselves because the
  // incoming Legion::Copy is const.
  streaming_gen_ = dez.unpack<std::optional<legate::detail::StreamingGeneration>>();
  machine_       = dez.unpack<Machine>();
  sharding_id_   = dez.unpack<std::uint32_t>();
  priority_      = dez.unpack<std::int32_t>();

  // Copy
  inputs_ = dez.unpack<legate::detail::SmallVector<Store>>();
  dez.next_requirement_list();
  outputs_ = dez.unpack<legate::detail::SmallVector<Store>>();
  dez.next_requirement_list();
  input_indirections_ = dez.unpack<legate::detail::SmallVector<Store>>();
  dez.next_requirement_list();
  output_indirections_ = dez.unpack<legate::detail::SmallVector<Store>>();
  for (auto&& input : inputs_) {
    LEGATE_ASSERT(!input.is_future());
  }
  for (auto&& output : outputs_) {
    LEGATE_ASSERT(!output.is_future());
  }
  for (auto&& input_indirection : input_indirections_) {
    LEGATE_ASSERT(!input_indirection.is_future());
  }
  for (auto&& output_indirection : output_indirections_) {
    LEGATE_ASSERT(!output_indirection.is_future());
  }
}

}  // namespace legate::mapping::detail
