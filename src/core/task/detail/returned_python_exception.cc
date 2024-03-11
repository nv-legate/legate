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

#include "core/task/detail/returned_python_exception.h"

#include "core/task/detail/exception.h"
#include "core/task/detail/returned_exception_common.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace legate::detail {

void ReturnedPythonException::legion_serialize(void* buffer) const
{
  auto rem_cap = legion_buffer_size();

  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, kind());
  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, size());
  static_cast<void>(pack_buffer(buffer, rem_cap, size(), static_cast<const char*>(data())));
}

void ReturnedPythonException::legion_deserialize(const void* buffer)
{
  // There is no information about the size of the buffer, nor can we know how much we need
  // until we unpack all of it. So we just lie and say we have infinite memory.
  auto rem_cap = std::numeric_limits<std::size_t>::max();
  ExceptionKind kind;

  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &kind);
  LegateAssert(kind == ExceptionKind::PYTHON);
  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &size_);
  if (const auto buf_size = size()) {
    // NOLINTNEXTLINE(readability-magic-numbers)
    static_assert(LEGATE_CPP_MIN_VERSION < 20, "Use make_unique_for_overwrite below");
    auto mem       = std::unique_ptr<char[]>{new char[buf_size]};
    const auto ptr = mem.get();

    static_cast<void>(unpack_buffer(buffer, buf_size, buf_size, &ptr));
    pickle_bytes_.reset(mem.release());
  }
}

ReturnValue ReturnedPythonException::pack() const
{
  const auto buffer_size = legion_buffer_size();
  const auto mem_kind    = find_memory_kind_for_executing_processor();
  auto buffer            = Legion::UntypedDeferredValue{buffer_size, mem_kind};
  const auto acc         = AccessorWO<std::int8_t, 1>{buffer, buffer_size, false};

  legion_serialize(acc.ptr(0));
  return {std::move(buffer), buffer_size};
}

std::string ReturnedPythonException::to_string() const
{
  std::stringstream ss;

  ss << "ReturnedPythonException(size = " << size() << ", bytes = ";
  for (std::uint64_t i = 0; i < size(); ++i) {
    ss << "\\x" << std::setw(2) << std::setfill('0') << std::hex
       << static_cast<std::int32_t>(pickle_bytes_.get()[i]);
  }
  ss << ')';
  return std::move(ss).str();
}

void ReturnedPythonException::throw_exception()
{
  throw PythonTaskException{
    std::exchange(size_, 0),
    const_pointer_cast<const char[]>(std::move(pickle_bytes_)).as_user_ptr()};
}

}  // namespace legate::detail
