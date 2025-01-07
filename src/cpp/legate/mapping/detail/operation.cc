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

#include "legate/mapping/detail/operation.h"

#include "legate/runtime/detail/library.h"
#include "legate/utilities/detail/core_ids.h"
#include "legate/utilities/detail/deserializer.h"
#include <legate/mapping/detail/mapping.h>

namespace legate::mapping::detail {

Mappable::Mappable(private_tag, MapperDataDeserializer dez)
  : machine_{dez.unpack<Machine>()},
    sharding_id_{dez.unpack<std::uint32_t>()},
    priority_{dez.unpack<std::int32_t>()}
{
}

Task::Task(const Legion::Task* task,
           Legion::Mapping::MapperRuntime* runtime,
           Legion::Mapping::MapperContext context)
  : Mappable{task}, task_{task}
{
  TaskDeserializer dez{task, runtime, context};
  library_             = dez.unpack<legate::detail::Library*>();
  inputs_              = dez.unpack_arrays();
  outputs_             = dez.unpack_arrays();
  reductions_          = dez.unpack_arrays();
  scalars_             = dez.unpack_scalars();
  can_raise_exception_ = dez.unpack<bool>();

  if (task_->tag ==
      static_cast<Legion::MappingTagID>(legate::detail::CoreMappingTag::TREE_REDUCE)) {
    inputs_.erase(
      std::remove_if(inputs_.begin(), inputs_.end(), [](const auto& inp) { return !inp->valid(); }),
      inputs_.end());
  }
}

LocalTaskID Task::task_id() const
{
  return library()->get_local_task_id(static_cast<GlobalTaskID>(task_->task_id));
}

Legion::VariantID Task::legion_task_variant() const
{
  return traits::detail::to_underlying(to_variant_code(target()));
}

// ==========================================================================================

Copy::Copy(const Legion::Copy* copy,
           Legion::Mapping::MapperRuntime* runtime,
           Legion::Mapping::MapperContext context)
  : copy_{copy}
{
  const auto reqs = {std::cref(copy->src_requirements),
                     std::cref(copy->dst_requirements),
                     std::cref(copy->src_indirect_requirements),
                     std::cref(copy->dst_indirect_requirements)};
  CopyDeserializer dez{copy, {reqs.begin(), reqs.end()}, runtime, context};

  // Mappable
  //
  // Cannot use Mappable ctor here because both the Mappable and Copy data live in the same
  // serdez buffer. So when Mappable() unpacks its stuff CopyDeserializer doesn't know that the
  // buffer should be advanced. And we cannot advance the pointer ourselves because the
  // incoming Legion::Copy is const.
  machine_     = dez.unpack<Machine>();
  sharding_id_ = dez.unpack<std::uint32_t>();
  priority_    = dez.unpack<std::int32_t>();

  // Copy
  inputs_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  outputs_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  input_indirections_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  output_indirections_ = dez.unpack<std::vector<Store>>();
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
