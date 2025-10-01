/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/core_mapper.h>

#include <legate/mapping/detail/machine.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/detail/operation.h>
#include <legate/mapping/operation.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/align.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/env_defaults.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/env.h>

#include <cstdlib>
#include <memory>
#include <vector>

namespace legate::mapping::detail {

// This is a custom mapper implementation that only has to map
// start-up tasks associated with Legate, no one else
// should be overriding this mapper so we burry it in here
class CoreMapper final : public Mapper {
 public:
  [[nodiscard]] std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override;
  [[nodiscard]] legate::Scalar tunable_value(legate::TunableID tunable_id) override;
  [[nodiscard]] std::optional<std::size_t> allocation_pool_size(
    const legate::mapping::Task& task, legate::mapping::StoreTarget memory_kind) override;
};

std::vector<legate::mapping::StoreMapping> CoreMapper::store_mappings(
  const legate::mapping::Task& task, const std::vector<StoreTarget>& /*options*/)
{
  if (task.task_id() == legate::LocalTaskID{legate::detail::CoreTask::OFFLOAD_TO}) {
    const auto mem_target = task.scalar(0).value<StoreTarget>();
    std::vector<mapping::StoreMapping> mappings;
    LEGATE_ASSERT(task.num_inputs() > 0);
    LEGATE_ASSERT(task.num_inputs() == task.num_outputs());
    // Offload task has the same inputs as outputs so mapping
    // just the inputs should be sufficient
    for (std::size_t i = 0; i < task.num_inputs(); ++i) {
      for (const auto& store : task.input(i).stores()) {
        mappings.emplace_back(mapping::StoreMapping::default_mapping(store, mem_target));
      }
    }
    LEGATE_ASSERT(!mappings.empty());
    return mappings;
  }

  return {};
}

Scalar CoreMapper::tunable_value(TunableID /*tunable_id*/)
{
  // Illegal tunable variable
  LEGATE_ABORT("Tunable values are no longer supported");
  return Scalar{0};
}

namespace {

// The INIT_NCCL task creates two buffers of type Payload that has two uint64_t-typed members, hence
// the following math for the pool size (the data type was crafted such that it will be 16-byte
// aligned).
constexpr std::size_t NCCL_WARMUP_BUFFER_SIZE = sizeof(std::uint64_t) * 2 * 2;

constexpr std::size_t EXTRA_SCALAR_ALIGNMENT = 16;

class AlignOfPointType {
 public:
  template <std::int32_t NDIM>
  std::size_t operator()() const
  {
    return alignof(Point<NDIM>);
  }
};

}  // namespace

std::optional<std::size_t> CoreMapper::allocation_pool_size(
  const legate::mapping::Task& task, legate::mapping::StoreTarget memory_kind)
{
  const auto task_id = legate::detail::to_underlying(task.task_id());

  // Python task workaround for allocatable tasks
  if (task_id >= legate::detail::CoreTask::FIRST_DYNAMIC_TASK) {
    return std::nullopt;
  }

  switch (task_id) {
    case legate::detail::CoreTask::EXTRACT_SCALAR: {
      // Extract scalar task doesn't use the framebuffer
      if (memory_kind == legate::mapping::StoreTarget::FBMEM) {
        return 0;
      }
      // The pool should be big enough to hold the source future and the extracted value
      const auto value_size  = task.scalar(1).value<std::size_t>();
      const auto future_size = task.scalar(2).value<std::size_t>();
      return legate::detail::round_up_to_multiple(future_size + value_size, EXTRA_SCALAR_ALIGNMENT);
    }
    case legate::detail::CoreTask::INIT_NCCL: {
      return legate::detail::Runtime::get_runtime().config().warmup_nccl() ? NCCL_WARMUP_BUFFER_SIZE
                                                                           : 0;
    }
    case legate::detail::CoreTask::FIND_BOUNDING_BOX: [[fallthrough]];
    case legate::detail::CoreTask::FIND_BOUNDING_BOX_SORTED: {
      if (memory_kind != legate::mapping::StoreTarget::FBMEM) {
        return 0;
      }
      auto&& type = task.input(0).type();
      const auto ndim =
        legate::is_rect_type(type) ? legate::ndim_rect_type(type) : legate::ndim_point_type(type);
      return legate::dim_dispatch(ndim, AlignOfPointType{}) * 2;
    }
    default: break;  // legate-lint: no-switch-default
  }

  LEGATE_ABORT(fmt::format("unhandled core task id: {}", task_id));
  return std::nullopt;
}

std::unique_ptr<Mapper> create_core_mapper() { return std::make_unique<CoreMapper>(); }

}  // namespace legate::mapping::detail
