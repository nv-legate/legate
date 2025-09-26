/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
  const auto kind     = runtime.get_executing_processor().kind();

  switch (kind) {
    case Processor::Kind::LOC_PROC: return Memory::Kind::SYSTEM_MEM;
    case Processor::Kind::TOC_PROC:
      return host_accessible ? Memory::Kind::Z_COPY_MEM : Memory::Kind::GPU_FB_MEM;
    case Processor::Kind::OMP_PROC:
      return runtime.local_machine().has_socket_memory() ? Memory::Kind::SOCKET_MEM
                                                         : Memory::Kind::SYSTEM_MEM;
    case Processor::Kind::PY_PROC: return Memory::Kind::SYSTEM_MEM;
    case Processor::Kind::NO_KIND: [[fallthrough]];
    case Processor::Kind::UTIL_PROC: [[fallthrough]];
    case Processor::Kind::IO_PROC: [[fallthrough]];
    case Processor::Kind::PROC_GROUP: [[fallthrough]];
    case Processor::Kind::PROC_SET: break;
  }
  LEGATE_ABORT("Unknown processor kind ", kind);
}

Memory find_memory_from_kind(Memory::Kind kind)
{
  auto&& runtime = detail::Runtime::get_runtime();

  return runtime.local_machine().get_memory(runtime.get_executing_processor(), kind);
}

}  // namespace legate
