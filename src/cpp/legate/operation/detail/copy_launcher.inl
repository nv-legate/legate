/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/copy_launcher.h>

namespace legate::detail {

template <bool SINGLE>
Legion::RegionRequirement CopyArg::create_requirement()
{
  return store_proj_.template create_requirement<SINGLE>(region_, {field_id_}, privilege_);
}

// ==========================================================================================
inline CopyLauncher::CopyLauncher(const mapping::detail::Machine& machine,
                                  std::int32_t priority,
                                  std::int64_t tag)
  : machine_{machine}, priority_{priority}, tag_{tag}
{
}

}  // namespace legate::detail
