/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/machine.h>

#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate {

mapping::StoreTarget find_store_target_for_executing_processor(bool host_accessible)
{
  if (!detail::has_started()) {
    throw detail::TracedException<std::runtime_error>{"Runtime has not started"};
  }

  const auto& runtime = detail::Runtime::get_runtime();

  if (runtime.config().need_cuda()) {
    return host_accessible ? mapping::StoreTarget::ZCMEM : mapping::StoreTarget::FBMEM;
  }
  return mapping::StoreTarget::SYSMEM;
}

}  // namespace legate
