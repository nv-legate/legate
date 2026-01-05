/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/task/detail/legion_task_body.h>

#include <legate/comm/communicator.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/detail/physical_array.h>
#include <legate/data/detail/physical_stores/future_physical_store.h>
#include <legate/data/detail/physical_stores/unbound_physical_store.h>
#include <legate/mapping/detail/machine.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/return_value.h>
#include <legate/task/detail/returned_exception.h>
#include <legate/task/detail/task.h>
#include <legate/task/detail/task_context.h>
#include <legate/task/detail/task_return.h>
#include <legate/task/task_context.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/deserializer.h>
#include <legate/utilities/typedefs.h>

#include <realm/cuda/cuda_module.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace legate::detail {

class Library;

namespace {

[[nodiscard]] TaskContext::CtorArgs make_task_context_ctor_args(
  const Legion::Task& task,
  VariantCode variant_kind,
  const std::vector<Legion::PhysicalRegion>& regions)
{
  TaskContext::CtorArgs ret;

  ret.variant_kind = variant_kind;

  TaskDeserializer dez{task, regions};

  static_cast<void>(dez.unpack<Library*>());
  static_cast<void>(dez.unpack<TaskInfo*>());
  ret.inputs     = dez.unpack_arrays();
  ret.outputs    = dez.unpack_arrays();
  ret.reductions = dez.unpack_arrays();
  ret.scalars    = dez.unpack_scalars();
  // The return future size is not used for task execution
  std::ignore                   = dez.unpack<std::size_t>();
  ret.can_raise_exception       = dez.unpack<bool>();
  ret.can_elide_device_ctx_sync = dez.unpack<bool>();

  bool insert_barrier = false;
  Legion::PhaseBarrier arrival{}, wait{};

  if (task.is_index_space) {
    insert_barrier = dez.unpack<bool>();
    if (insert_barrier) {
      arrival = dez.unpack<Legion::PhaseBarrier>();
      wait    = dez.unpack<Legion::PhaseBarrier>();
    }
    ret.comms = dez.unpack<SmallVector<comm::Communicator>>();
  }

  // For reduction tree cases, some input stores may be mapped to NO_REGION
  // when the number of subregions isn't a multiple of the chosen radix.
  // To simplify the programming mode, we filter out those "invalid" stores out.
  if (task.tag == static_cast<Legion::MappingTagID>(CoreMappingTag::TREE_REDUCE)) {
    constexpr auto is_invalid_array = [](const InternalSharedPtr<PhysicalArray>& inp) {
      return !inp->valid();
    };

    ret.inputs.erase(std::remove_if(ret.inputs.begin(), ret.inputs.end(), is_invalid_array),
                     ret.inputs.end());
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
  return ret;
}

}  // namespace

LegionTaskContext::LegionTaskContext(const Legion::Task& legion_task,
                                     VariantCode variant_kind,
                                     const std::vector<Legion::PhysicalRegion>& regions,
                                     mapping::detail::Machine&& machine)
  : TaskContext{make_task_context_ctor_args(legion_task, variant_kind, regions)},
    task_{legion_task},
    machine_{std::move(machine)}
{
  // If the task is running on a GPU, AND there is at least one scalar store for reduction,
  // then we need to wait for all the host-to-device copies for initialization to finish,
  // UNLESS the user has promised to use the task stream. In that case we can skip this sync.
  constexpr auto is_scalar_store = [](const InternalSharedPtr<PhysicalArray>& array) -> bool {
    return dynamic_cast<const FuturePhysicalStore*>(array->data().get());
  };
  if (LEGATE_DEFINED(LEGATE_USE_CUDA) && !can_elide_device_ctx_sync() &&
      (legion_task_().current_proc.kind() == Processor::Kind::TOC_PROC) &&
      std::any_of(reductions().begin(), reductions().end(), is_scalar_store)) {
    cuda::detail::get_cuda_driver_api()->stream_synchronize(
      Runtime::get_runtime().get_cuda_stream());
  }
}

const Legion::Task& LegionTaskContext::legion_task_() const noexcept { return task_; }

std::vector<ReturnValue> LegionTaskContext::get_return_values_() const
{
  std::vector<ReturnValue> return_values;

  return_values.reserve(get_scalar_stores_().size() + can_raise_exception());

  for (auto&& store : get_scalar_stores_()) {
    return_values.push_back(store->as_future_store().pack());
  }
  // If this is a reduction task, we do sanity checks on the invariants
  // the Python code relies on.
  if (legion_task_().tag == static_cast<Legion::MappingTagID>(CoreMappingTag::TREE_REDUCE)) {
    if (get_unbound_stores_().size() != 1) {
      LEGATE_ABORT("Reduction tasks must have only one unbound output and no others");
    }
  }

  return return_values;
}

// ==========================================================================================

LegionTaskContext::LegionTaskContext(const Legion::Task& legion_task,
                                     VariantCode variant_kind,
                                     const std::vector<Legion::PhysicalRegion>& regions)
  // WARNING: if the Machine is no longer the second thing to be deserialized, then this will break
  : LegionTaskContext{legion_task, variant_kind, regions, [&] {
                        auto dez = mapping::detail::MapperDataDeserializer{legion_task};

                        static_cast<void>(dez.unpack<std::optional<StreamingGeneration>>());
                        return dez.unpack<mapping::detail::Machine>();
                      }()}
{
}

GlobalTaskID LegionTaskContext::task_id() const noexcept
{
  return static_cast<GlobalTaskID>(legion_task_().task_id);
}

bool LegionTaskContext::is_single_task() const noexcept { return !legion_task_().is_index_space; }

const DomainPoint& LegionTaskContext::get_task_index() const noexcept
{
  return legion_task_().index_point;
}

const Domain& LegionTaskContext::get_launch_domain() const noexcept
{
  return legion_task_().index_domain;
}

std::string_view LegionTaskContext::get_provenance() const
{
  return legion_task_().get_provenance_string();
}

const mapping::detail::Machine& LegionTaskContext::machine() const noexcept { return machine_; }

TaskReturn LegionTaskContext::pack_return_values(const std::optional<ReturnedException>& exn) const
{
  auto return_values = get_return_values_();

  if (can_raise_exception()) {
    return_values.push_back(exn.value_or(ReturnedException{}).pack());
  }
  return TaskReturn{std::move(return_values)};
}

// ==========================================================================================

void legion_task_body(VariantImpl variant_impl,
                      VariantCode variant_kind,
                      std::optional<std::string_view> task_name,
                      const void* args,
                      std::size_t arglen,
                      Processor p)
{
  // Legion preamble
  const Legion::Task* task{};
  const std::vector<Legion::PhysicalRegion>* regions{};
  Legion::Context legion_context{};
  Legion::Runtime* runtime{};

  Legion::Runtime::legion_task_preamble(args, arglen, p, task, regions, legion_context, runtime);

  const auto get_task_name = [&] {
    if (!task_name.has_value()) {
      // task->get_task_name() returns a const char * (which requires a call to strlen() to
      // convert to string_view), so do it once, and cache it for later.
      task_name = task->get_task_name();
    }
    return *task_name;
  };

  const auto nvtx_range =
    task_detail::make_nvtx_range(get_task_name, [&] { return task->get_provenance_string(); });
  static_cast<void>(nvtx_range);

  show_progress(task, legion_context, runtime);

  auto legion_task_ctx = LegionTaskContext{*task, variant_kind, *regions};

  if constexpr (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    Realm::Cuda::set_task_ctxsync_required(!legion_task_ctx.can_elide_device_ctx_sync());
  }

  const auto exception =
    task_detail::task_body(legate::TaskContext{&legion_task_ctx}, variant_impl, get_task_name);

  const auto return_values = legion_task_ctx.pack_return_values(exception);

  // Legion postamble
  return_values.finalize(legion_context, legion_task_ctx.can_elide_device_ctx_sync());
}

}  // namespace legate::detail
