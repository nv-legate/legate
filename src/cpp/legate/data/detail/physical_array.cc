/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/physical_array.h>

#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

namespace legate::detail {

const InternalSharedPtr<PhysicalStore>& PhysicalArray::data() const
{
  throw TracedException<std::invalid_argument>{"Data store of a nested array cannot be retrieved"};
}

}  // namespace legate::detail
