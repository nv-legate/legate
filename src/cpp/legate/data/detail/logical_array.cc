/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_array.h>

#include <legate/data/detail/logical_arrays/base_logical_array.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate::detail {

const InternalSharedPtr<LogicalStore>& LogicalArray::data() const
{
  throw TracedException<std::invalid_argument>{"Data store of a nested array cannot be retrieved"};
}

/*static*/ InternalSharedPtr<LogicalArray> LogicalArray::from_store(
  InternalSharedPtr<LogicalStore> store)
{
  return make_internal_shared<BaseLogicalArray>(std::move(store));
}

}  // namespace legate::detail
