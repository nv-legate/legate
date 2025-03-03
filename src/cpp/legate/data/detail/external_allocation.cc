/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/external_allocation.h>

namespace legate::detail {

void ExternalAllocation::maybe_deallocate()
{
  if (deleter_) {
    (*deleter_)(ptr_);
    deleter_.reset();
  }
}

}  // namespace legate::detail
