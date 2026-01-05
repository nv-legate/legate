/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/logical_array.h>

namespace legate::detail {

inline bool LogicalArray::needs_flush() const
{
  // TODO(wonchanl): We will eventually need to handle unbound stores in the deferred manner
  return unbound() || is_mapped();
}

}  // namespace legate::detail
