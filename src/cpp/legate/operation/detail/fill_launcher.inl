/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/operation/detail/fill_launcher.h>

namespace legate::detail {

inline FillLauncher::FillLauncher(const mapping::detail::Machine& machine,
                                  std::int32_t priority,
                                  std::string_view provenance)
  : machine_{machine}, priority_{priority}, provenance_{provenance}
{
}

}  // namespace legate::detail
