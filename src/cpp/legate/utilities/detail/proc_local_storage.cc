/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/proc_local_storage.h>

#include <legate/utilities/detail/traced_exception.h>

#include <legion/runtime.h>
#include <realm/id.h>

#include <fmt/format.h>
#include <fmt/std.h>

#include <cstddef>
#include <stdexcept>
#include <typeinfo>

namespace legate::detail {

std::size_t processor_id()
{
  constexpr std::size_t LOCAL_PROC_BITWIDTH = Realm::ID::FMT_Processor::proc_idx::BITS;
  constexpr std::size_t LOCAL_PROC_MASK     = (1 << LOCAL_PROC_BITWIDTH) - 1;
  // Processor IDs are numbered locally in each node and local indices are encoded in the LSBs, so
  // here we mask out the rest to get the rank-local index of the processor
  const auto proc =
    Legion::Runtime::get_runtime()->get_executing_processor(Legion::Runtime::get_context());

  return static_cast<std::size_t>(proc.id & LOCAL_PROC_MASK);
}

void throw_invalid_proc_local_storage_access(const std::type_info& value_type)
{
  const auto proc =
    Legion::Runtime::get_runtime()->get_executing_processor(Legion::Runtime::get_context());

  throw TracedException<std::logic_error>{
    fmt::format("Processor local storage of type {} hasn't been initialized for processor {:x}. "
                "Please use `.emplace()` to initialize the storage first.",
                value_type,
                proc.id)};
}

}  // namespace legate::detail
