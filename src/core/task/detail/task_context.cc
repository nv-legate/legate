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

#include "core/task/detail/task_context.h"

#include "core/data/detail/store.h"
#include "core/utilities/deserializer.h"

#ifdef LEGATE_USE_CUDA
#include "core/cuda/cuda_help.h"
#endif

namespace legate::detail {

TaskContext::TaskContext(const Legion::Task* task,
                         const std::vector<Legion::PhysicalRegion>& regions)
  : task_(task), regions_(regions)
{
  {
    mapping::MapperDataDeserializer dez(task);
    machine_ = dez.unpack<mapping::detail::Machine>();
  }

  TaskDeserializer dez(task, regions);
  inputs_     = dez.unpack<std::vector<legate::Store>>();
  outputs_    = dez.unpack<std::vector<legate::Store>>();
  reductions_ = dez.unpack<std::vector<legate::Store>>();
  scalars_    = dez.unpack<std::vector<legate::Scalar>>();

  // Make copies of stores that we need to postprocess, as clients might move the stores away
  for (auto& output : outputs_) {
    if (output.is_unbound_store()) {
      unbound_stores_.push_back(output);
    } else if (output.is_future()) {
      scalar_stores_.push_back(output);
    }
  }
  for (auto& reduction : reductions_) {
    if (reduction.is_future()) { scalar_stores_.push_back(reduction); }
  }

  can_raise_exception_ = dez.unpack<bool>();

  bool insert_barrier = false;
  Legion::PhaseBarrier arrival, wait;
  if (task->is_index_space) {
    insert_barrier = dez.unpack<bool>();
    if (insert_barrier) {
      arrival = dez.unpack<Legion::PhaseBarrier>();
      wait    = dez.unpack<Legion::PhaseBarrier>();
    }
    comms_ = dez.unpack<std::vector<comm::Communicator>>();
  }

  // For reduction tree cases, some input stores may be mapped to NO_REGION
  // when the number of subregions isn't a multiple of the chosen radix.
  // To simplify the programming mode, we filter out those "invalid" stores out.
  if (task_->tag == LEGATE_CORE_TREE_REDUCE_TAG) {
    std::vector<legate::Store> inputs;
    for (auto& input : inputs_)
      if (input.valid()) inputs.push_back(std::move(input));
    inputs_.swap(inputs);
  }

  // CUDA drivers < 520 have a bug that causes deadlock under certain circumstances
  // if the application has multiple threads that launch blocking kernels, such as
  // NCCL all-reduce kernels. This barrier prevents such deadlock by making sure
  // all CUDA driver calls from Realm are done before any of the GPU tasks starts
  // making progress.
  if (insert_barrier) {
    arrival.arrive();
    wait.wait();
  }
#ifdef LEGATE_USE_CUDA
  // If the task is running on a GPU and there is at least one scalar store for reduction,
  // we need to wait for all the host-to-device copies for initialization to finish
  if (Processor::get_executing_processor().kind() == Processor::Kind::TOC_PROC)
    for (auto& reduction : reductions_)
      if (reduction.is_future()) {
        CHECK_CUDA(cudaDeviceSynchronize());
        break;
      }
#endif
}

void TaskContext::make_all_unbound_stores_empty()
{
  for (auto& output : outputs_)
    if (output.is_unbound_store()) output.bind_empty_data();
}

ReturnValues TaskContext::pack_return_values() const
{
  auto return_values = get_return_values();
  if (can_raise_exception_) {
    ReturnedException exn{};
    return_values.push_back(exn.pack());
  }
  return ReturnValues(std::move(return_values));
}

ReturnValues TaskContext::pack_return_values_with_exception(int32_t index,
                                                            const std::string& error_message) const
{
  auto return_values = get_return_values();
  if (can_raise_exception_) {
    ReturnedException exn(index, error_message);
    return_values.push_back(exn.pack());
  }
  return ReturnValues(std::move(return_values));
}

std::vector<ReturnValue> TaskContext::get_return_values() const
{
  std::vector<ReturnValue> return_values;

  for (auto& store : unbound_stores_) { return_values.push_back(store.impl()->pack_weight()); }
  for (auto& store : scalar_stores_) { return_values.push_back(store.impl()->pack()); }

  // If this is a reduction task, we do sanity checks on the invariants
  // the Python code relies on.
  if (task_->tag == LEGATE_CORE_TREE_REDUCE_TAG) {
    if (return_values.size() != 1 || unbound_stores_.size() != 1) {
      legate::log_legate.error("Reduction tasks must have only one unbound output and no others");
      LEGATE_ABORT;
    }
  }

  return return_values;
}

}  // namespace legate::detail
