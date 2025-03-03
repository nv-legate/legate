/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/proc_local_storage.h>

#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>
#include <fmt/std.h>

#include <stdexcept>
#include <typeinfo>

namespace legate::detail {

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
