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

Mappable::Mappable() {}

Mappable::Mappable(const Legion::Mappable* mappable)
{
  MapperDataDeserializer dez(mappable);
  machine_     = dez.unpack<Machine>();
  sharding_id_ = dez.unpack<uint32_t>();
}

Task::Task(const Legion::Task* task,
           const legate::detail::Library* library,
           Legion::Mapping::MapperRuntime* runtime,
           const Legion::Mapping::MapperContext context)
  : Mappable(task), task_(task), library_(library)
{
  TaskDeserializer dez(task, runtime, context);
  inputs_     = dez.unpack_arrays();
  outputs_    = dez.unpack_arrays();
  reductions_ = dez.unpack_arrays();
  scalars_    = dez.unpack_scalars();
}

int64_t Task::task_id() const { return library_->get_local_task_id(task_->task_id); }

TaskTarget Task::target() const
{
  switch (task_->target_proc.kind()) {
    case Processor::LOC_PROC: return TaskTarget::CPU;
    case Processor::TOC_PROC: return TaskTarget::GPU;
    case Processor::OMP_PROC: return TaskTarget::OMP;
    default: {
      assert(false);
    }
  }
  assert(false);
  return TaskTarget::CPU;
}

Copy::Copy(const Legion::Copy* copy,
           Legion::Mapping::MapperRuntime* runtime,
           const Legion::Mapping::MapperContext context)
  : Mappable(), copy_(copy)
{
  CopyDeserializer dez(copy,
                       {copy->src_requirements,
                        copy->dst_requirements,
                        copy->src_indirect_requirements,
                        copy->dst_indirect_requirements},
                       runtime,
                       context);
  machine_     = dez.unpack<Machine>();
  sharding_id_ = dez.unpack<uint32_t>();
  inputs_      = dez.unpack<Stores>();
  dez.next_requirement_list();
  outputs_ = dez.unpack<Stores>();
  dez.next_requirement_list();
  input_indirections_ = dez.unpack<Stores>();
  dez.next_requirement_list();
  output_indirections_ = dez.unpack<Stores>();
#ifdef DEBUG_LEGATE
  for (auto& input : inputs_) assert(!input.is_future());
  for (auto& output : outputs_) assert(!output.is_future());
  for (auto& input_indirection : input_indirections_) assert(!input_indirection.is_future());
  for (auto& output_indirection : output_indirections_) assert(!output_indirection.is_future());
#endif
}

}  // namespace legate::mapping::detail
