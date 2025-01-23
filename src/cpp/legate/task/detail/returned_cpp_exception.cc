/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/task/detail/returned_cpp_exception.h"

#include "legate/task/exception.h"
#include "legate/utilities/detail/formatters.h"
#include <legate/task/detail/returned_exception.h>
#include <legate/utilities/detail/traced_exception.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace legate::detail {

void ReturnedCppException::throw_exception()
{
  // Don't wrap this in a trace, it may already contain a traced exception.
  throw TaskException{std::exchange(index_, 0), std::move(message_)};  // legate-lint: no-trace
}

// Note, this function returns an upper bound on the size of the type as it also incorporates
// alignment requirements for each member. It cannot know how much of the extra alignment
// padding it needs because that depends on how the input pointer is aligned when it goes to
// pack.
std::size_t ReturnedCppException::legion_buffer_size() const
{
  auto size =
    max_aligned_size_for_type<decltype(kind())>() + max_aligned_size_for_type<decltype(raised())>();

  if (raised()) {
    size += max_aligned_size_for_type<decltype(index())>();
    size += max_aligned_size_for_type<decltype(this->size())>();  // size of string
    size += this->size();  // the actual string (not null terminated!)
  }
  return size;
}

void ReturnedCppException::legion_serialize(void* buffer) const
{
  auto rem_cap = legion_buffer_size();

  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, kind());
  std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, raised());
  if (raised()) {
    std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, index());
    std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, size());
    std::tie(buffer, rem_cap) =
      pack_buffer(buffer,
                  rem_cap,
                  size(),
                  // We pass the size
                  message().data()  // NOLINT(bugprone-suspicious-stringview-data-usage)
      );
  }
}

void ReturnedCppException::legion_deserialize(const void* buffer)
{
  ExceptionKind kind;
  bool raised;
  // There is no information about the size of the buffer, nor can we know how much we need
  // until we unpack all of it. So we just lie and say we have infinite memory.
  auto rem_cap = std::numeric_limits<std::size_t>::max();

  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &kind);
  LEGATE_ASSERT(kind == ExceptionKind::CPP);
  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &raised);
  if (raised) {
    std::uint64_t mess_size;

    std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &index_);
    std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &mess_size);
    if (rem_cap < mess_size) {
      throw TracedException<std::range_error>{
        fmt::format("Remaining capacity of serdez buffer: {} < length of stored string: {}. This "
                    "indicates a bug in the packing routine",
                    rem_cap,
                    mess_size)};
    }
    message_ = std::string{static_cast<const char*>(buffer), mess_size};
  }
}

ReturnValue ReturnedCppException::pack() const
{
  const auto buffer_size = legion_buffer_size();

  if (buffer_size > ReturnedException::max_size()) {
    throw TracedException<std::runtime_error>{
      fmt::format("The size of raised exception ({}) exceeds the maximum number of exception ({}). "
                  "Please increase the value for LEGATE_MAX_EXCEPTION_SIZE.",
                  buffer_size,
                  ReturnedException::max_size())};
  }

  const auto mem_kind = find_memory_kind_for_executing_processor();
  auto buffer         = Legion::UntypedDeferredValue{buffer_size, mem_kind};
  const auto acc      = AccessorWO<std::int8_t, 1>{buffer, buffer_size, false};

  legion_serialize(acc.ptr(0));
  // No alignment for returned exceptions, as they are always memcpy-ed
  return {std::move(buffer), buffer_size, 1 /*alignment*/};
}

std::string ReturnedCppException::to_string() const
{
  return fmt::format("ReturnedCppException(index = {}, message = {})", index(), message());
}

}  // namespace legate::detail
