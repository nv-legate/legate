/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/timing.h>

#include <legate/utilities/abort.h>
#include <legate/utilities/detail/type_traits.h>

namespace legate::detail {

void Timing::launch()
{
  // TODO(wonchanl): Need a v2 implementation
}

void Timing::launch(Strategy* /*strategy*/) { launch(); }

}  // namespace legate::detail
