/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/machine.h>

#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/detail/traced_exception.h>

#include <stdexcept>

namespace legate {

Memory::Kind find_memory_kind_for_executing_processor(bool host_accessible)
{
  if (!Legion::Runtime::has_runtime()) {
    throw detail::TracedException<std::runtime_error>{"Runtime has not started"};
  }

  const auto& runtime = detail::Runtime::get_runtime();

  if (runtime.config().need_cuda()) {
    return host_accessible ? Memory::Kind::Z_COPY_MEM : Memory::Kind::GPU_FB_MEM;
  }
  return Memory::Kind::SYSTEM_MEM;
}

}  // namespace legate
