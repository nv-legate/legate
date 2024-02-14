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

#include "core/mapping/detail/operation.h"

#include "core/runtime/detail/library.h"
#include "core/utilities/deserializer.h"

namespace legate::mapping::detail {

Mappable::Mappable(private_tag, MapperDataDeserializer dez)
  : machine_{dez.unpack<Machine>()}, sharding_id_{dez.unpack<std::uint32_t>()}
{
}

Task::Task(const Legion::Task* task,
           const legate::detail::Library* library,
           Legion::Mapping::MapperRuntime* runtime,
           Legion::Mapping::MapperContext context)
  : Mappable{task}, library_{library}, task_{task}
{
  TaskDeserializer dez{task, runtime, context};
  inputs_     = dez.unpack_arrays();
  outputs_    = dez.unpack_arrays();
  reductions_ = dez.unpack_arrays();
  scalars_    = dez.unpack_scalars();
}

std::int64_t Task::task_id() const { return library_->get_local_task_id(task_->task_id); }

TaskTarget Task::target() const
{
  switch (const auto kind = task_->target_proc.kind()) {
    case Processor::LOC_PROC: return TaskTarget::CPU;
    case Processor::TOC_PROC: return TaskTarget::GPU;
    case Processor::OMP_PROC: return TaskTarget::OMP;
    default: throw std::invalid_argument{"Invalid task target: " + std::to_string(kind)};
  }
}

Copy::Copy(const Legion::Copy* copy,
           Legion::Mapping::MapperRuntime* runtime,
           Legion::Mapping::MapperContext context)
  : copy_{copy}
{
  CopyDeserializer dez{copy,
                       {copy->src_requirements,
                        copy->dst_requirements,
                        copy->src_indirect_requirements,
                        copy->dst_indirect_requirements},
                       runtime,
                       context};
  machine_     = dez.unpack<Machine>();
  sharding_id_ = dez.unpack<std::uint32_t>();
  inputs_      = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  outputs_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  input_indirections_ = dez.unpack<std::vector<Store>>();
  dez.next_requirement_list();
  output_indirections_ = dez.unpack<std::vector<Store>>();
  for (auto& input : inputs_) {
    LegateAssert(!input.is_future());
  }
  for (auto& output : outputs_) {
    LegateAssert(!output.is_future());
  }
  for (auto& input_indirection : input_indirections_) {
    LegateAssert(!input_indirection.is_future());
  }
  for (auto& output_indirection : output_indirections_) {
    LegateAssert(!output_indirection.is_future());
  }
}

}  // namespace legate::mapping::detail
