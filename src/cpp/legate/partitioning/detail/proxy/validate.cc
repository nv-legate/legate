/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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

namespace legate::detail::proxy {

void ValidateVisitor::operator()(const legate::proxy::ArrayArgument& array) const
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
    case legate::proxy::ArrayArgument::Kind::INPUT: {
      check_array_index(array.index, signature.inputs(), "input");
      break;
    }
    case legate::proxy::ArrayArgument::Kind::OUTPUT: {
      check_array_index(array.index, signature.outputs(), "output");
      break;
    }
    case legate::proxy::ArrayArgument::Kind::REDUCTION: {
      check_array_index(array.index, signature.redops(), "reduction");
      break;
    }
  }
}

}  // namespace legate::detail::proxy
