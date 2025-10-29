/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/task_array_arg.h>

#include <legate/data/detail/logical_array.h>
#include <legate/operation/projection.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <optional>
#include <utility>
#include <variant>

namespace legate::detail {

TaskArrayArg::TaskArrayArg(Legion::PrivilegeMode priv,
                           InternalSharedPtr<LogicalArray> _array,
                           std::optional<SymbolicPoint> _projection)
  : privilege{priv}, array{std::move(_array)}, projection{std::move(_projection)}
{
  // These objects should only be constructed from add_input(), add_output(), or
  // add_reduction(), so the incoming privilege should only ever be these base privileges. The
  // privilege coalescing code (and by extension streaming code) assumes this is the case, with
  // the only addition being that these privileges are additionally fixed up with
  // LEGION_DISCARD_OUTPUT_MASK.
  LEGATE_ASSERT(privilege == LEGION_READ_ONLY || privilege == LEGION_WRITE_ONLY ||
                privilege == LEGION_REDUCE);
}

TaskArrayArg::TaskArrayArg(Legion::PrivilegeMode priv, InternalSharedPtr<PhysicalArray> _array)
  : privilege{priv}, array{std::move(_array)}
{
  // These objects should only be constructed from add_input(), add_output(), or
  // add_reduction(), so the incoming privilege should only ever be these base privileges. The
  // privilege coalescing code (and by extension streaming code) assumes this is the case, with
  // the only addition being that these privileges are additionally fixed up with
  // LEGION_DISCARD_OUTPUT_MASK.
  LEGATE_ASSERT(privilege == LEGION_READ_ONLY || privilege == LEGION_WRITE_ONLY ||
                privilege == LEGION_REDUCE);
}

bool TaskArrayArg::needs_flush() const
{
  return std::visit(
    Overload{[](const InternalSharedPtr<LogicalArray>& arr) -> bool { return arr->needs_flush(); },
             [](const InternalSharedPtr<PhysicalArray>&) -> bool {
               return false;  // PhysicalArray doesn't need flush
             }},
    array);
}

}  // namespace legate::detail
