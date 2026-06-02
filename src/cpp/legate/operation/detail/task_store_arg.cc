/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/task_store_arg.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/physical_store.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <optional>
#include <utility>
#include <variant>

namespace legate::detail {

TaskStoreArg::TaskStoreArg(Legion::PrivilegeMode priv,
                           InternalSharedPtr<LogicalStore> _store,
                           const Variable* _variable)
  : privilege{priv}, store{std::move(_store)}, variable{_variable}
{
  // These objects should only be constructed from add_input(), add_output(), or
  // add_reduction(), so the incoming privilege should only ever be these base privileges. The
  // privilege coalescing code (and by extension streaming code) assumes this is the case, with
  // the only addition being that these privileges are additionally fixed up with
  // LEGION_DISCARD_OUTPUT_MASK.
  LEGATE_ASSERT(privilege == LEGION_READ_ONLY || privilege == LEGION_WRITE_ONLY ||
                privilege == LEGION_REDUCE);
}

TaskStoreArg::TaskStoreArg(Legion::PrivilegeMode priv, InternalSharedPtr<PhysicalStore> _store)
  : privilege{priv}, store{std::move(_store)}
{
  // These objects should only be constructed from add_input(), add_output(), or
  // add_reduction(), so the incoming privilege should only ever be these base privileges. The
  // privilege coalescing code (and by extension streaming code) assumes this is the case, with
  // the only addition being that these privileges are additionally fixed up with
  // LEGION_DISCARD_OUTPUT_MASK.
  LEGATE_ASSERT(privilege == LEGION_READ_ONLY || privilege == LEGION_WRITE_ONLY ||
                privilege == LEGION_REDUCE);
}

bool TaskStoreArg::needs_flush() const
{
  return std::visit(
    Overload{[](const InternalSharedPtr<LogicalStore>& st) -> bool { return st->needs_flush(); },
             [](const InternalSharedPtr<PhysicalStore>&) -> bool {
               return false;  // PhysicalStore doesn't need flush
             }},
    store);
}

}  // namespace legate::detail
