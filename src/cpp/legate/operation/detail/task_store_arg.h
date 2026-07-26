/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/internal_shared_ptr.h>

#include <optional>

namespace legate::detail {

class LogicalStore;
class Variable;

class TaskStoreArg {
 public:
  /**
   * @brief Construct a TaskStoreArg for LogicalStore.
   *
   * @param _store The LogicalStore for this argument.
   * @param _variable The store's partition variable.
   */
  TaskStoreArg(InternalSharedPtr<LogicalStore> _store, const Variable* _variable = nullptr);

  InternalSharedPtr<LogicalStore> store{};
  const Variable* variable{};
};

}  // namespace legate::detail

#include <legate/operation/detail/task_store_arg.inl>
