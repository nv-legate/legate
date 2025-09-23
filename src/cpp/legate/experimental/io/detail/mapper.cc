/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/io/detail/mapper.h>

#include <legate/data/buffer.h>
#include <legate/mapping/operation.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/align.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/type_traits.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace legate::experimental::io::detail {

std::vector<mapping::StoreMapping> Mapper::store_mappings(
  const mapping::Task& task, const std::vector<mapping::StoreTarget>& options)
{
  const auto target = options.front();

  switch (task.task_id()) {
    case legate::LocalTaskID{legate::detail::CoreTask::IO_HDF5_FILE_READ}: [[fallthrough]];
    case legate::LocalTaskID{legate::detail::CoreTask::IO_HDF5_FILE_WRITE_VDS}: [[fallthrough]];
    case legate::LocalTaskID{legate::detail::CoreTask::IO_HDF5_FILE_COMBINE_VDS}: {
      switch (target) {
        case legate::mapping::StoreTarget::SYSMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::SOCKETMEM:
          // Host-only tasks will use direct strided reads/writes
          return {};
        case legate::mapping::StoreTarget::FBMEM: [[fallthrough]];
        case legate::mapping::StoreTarget::ZCMEM:
          if (legate::detail::Runtime::get_runtime().config().io_use_vfd_gds()) {
            // GPU tasks with GDS will also use strided reads/writes
            return {};
          }
          // GPU tasks without GDS need to use a temporary buffer to stage the reads/writes and
          // therefore need contiguous buffers
          break;
      }
    } break;
    default: break;  // legate-lint: no-switch-default
  }

  // Require that all tasks get their Stores in contiguous buffers
  std::vector<mapping::StoreMapping> mappings;

  mappings.reserve(task.num_inputs() + task.num_outputs());
  for (std::size_t i = 0; i < task.num_inputs(); ++i) {
    auto&& store = task.input(static_cast<std::uint32_t>(i)).data();

    mappings.emplace_back(mapping::StoreMapping::default_mapping(store, target, /* exact */ true));
  }
  for (std::size_t i = 0; i < task.num_outputs(); ++i) {
    auto&& store = task.output(static_cast<std::uint32_t>(i)).data();

    mappings.emplace_back(mapping::StoreMapping::default_mapping(store, target, /* exact */ true));
  }
  return mappings;
}

namespace {

/**
 * @brief Computes the HDF5 task's Legion pool size.
 *
 * @param array The input or output array being written to disk.
 *
 * @return The task pool size.
 */
[[nodiscard]] std::size_t hdf5_task_pool_size(const legate::mapping::Array& array)
{
  const auto bytes = array.domain().get_volume() * array.type().size();

  return legate::detail::round_up_to_multiple(bytes, DEFAULT_ALIGNMENT);
}

}  // namespace

std::optional<std::size_t> Mapper::allocation_pool_size(const mapping::Task& task,
                                                        mapping::StoreTarget memory_kind)
{
  // Bounce buffers are created only on the zero-copy memory
  if (memory_kind != mapping::StoreTarget::ZCMEM) {
    return 0;
  }

  // Bounce bffers are not created when GDS is on
  if (legate::detail::Runtime::get_runtime().config().io_use_vfd_gds()) {
    return 0;
  }

  const auto task_id = task.task_id();

  switch (task_id) {
    case legate::LocalTaskID{legate::detail::CoreTask::IO_HDF5_FILE_READ}:
      return hdf5_task_pool_size(task.output(0));
    case legate::LocalTaskID{legate::detail::CoreTask::IO_HDF5_FILE_WRITE_VDS}:
      return hdf5_task_pool_size(task.input(0));
  }
  LEGATE_ABORT("Unhandled task id ", legate::detail::to_underlying(task_id));
}

legate::Scalar Mapper::tunable_value(TunableID)
{
  LEGATE_CHECK(false);
  return legate::Scalar{0};
}

}  // namespace legate::experimental::io::detail
