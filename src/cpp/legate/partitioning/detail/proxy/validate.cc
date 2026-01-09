/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/proxy/validate.h>

#include <legate/partitioning/detail/proxy/constraint.h>
#include <legate/partitioning/proxy.h>
#include <legate/task/detail/task_signature.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <cstdint>
#include <optional>
#include <string_view>

namespace legate::detail {

void ValidateVisitor::operator()(const ProxyArrayArgument& array) const
{
  const auto check_array_index = [&](std::uint32_t array_index,
                                     const std::optional<TaskSignature::Nargs>& nargs,
                                     std::string_view arg_kind) {
    if (!nargs.has_value() || nargs->compatible_with(array_index, /* strict */ false)) {
      return;
    }

    throw TracedException<std::out_of_range>{
      fmt::format("Invalid task signature for task {}. {} argument index {} (for {} constraint) is "
                  "out of range for specified signature: {}",
                  task_name,
                  arg_kind,
                  array_index,
                  constraint.name(),
                  *nargs)};
  };

  switch (array.kind) {
    case ProxyArrayArgument::Kind::INPUT: {
      check_array_index(array.index, signature.inputs(), "input");
      break;
    }
    case ProxyArrayArgument::Kind::OUTPUT: {
      check_array_index(array.index, signature.outputs(), "output");
      break;
    }
    case ProxyArrayArgument::Kind::REDUCTION: {
      check_array_index(array.index, signature.redops(), "reduction");
      break;
    }
  }
}

}  // namespace legate::detail
