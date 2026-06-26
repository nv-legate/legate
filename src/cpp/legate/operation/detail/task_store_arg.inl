/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/task_store_arg.h>
#include <legate/utilities/internal_shared_ptr.h>

namespace legate::detail {

inline TaskStoreArg::TaskStoreArg(InternalSharedPtr<LogicalStore> _store, const Variable* _variable)
  : store{std::move(_store)}, variable{_variable}
{
}

inline bool TaskStoreArg::needs_flush() const { return store->needs_flush(); }

}  // namespace legate::detail
