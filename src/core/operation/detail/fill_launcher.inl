/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/operation/detail/fill_launcher.h"

namespace legate::detail {

inline FillLauncher::FillLauncher(const mapping::detail::Machine& machine,
                                  std::int32_t priority,
                                  std::int64_t tag)
  : machine_{machine}, priority_{priority}, tag_{tag}
{
  static_cast<void>(tag_);
}

}  // namespace legate::detail
