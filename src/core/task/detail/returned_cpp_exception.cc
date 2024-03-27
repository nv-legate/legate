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

#include "core/task/detail/returned_cpp_exception.h"

#include "core/task/exception.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace legate::detail {

void ReturnedCppException::throw_exception()
{
  throw TaskException{std::exchange(index_, 0), std::move(message_)};
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
    std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, size(), message_.data());
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
  LegateAssert(kind == ExceptionKind::CPP);
  std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &raised);
  if (raised) {
    std::uint64_t mess_size;

    std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &index_);
    std::tie(buffer, rem_cap) = unpack_buffer(buffer, rem_cap, &mess_size);
    if (rem_cap < mess_size) {
      std::stringstream ss;

      ss << "Remaining capacity of serdez buffer: " << rem_cap
         << " < length of stored string: " << mess_size
         << ". This indicates a bug in the packing routine!";
      throw std::range_error{std::move(ss).str()};
    }
    message_ = std::string{static_cast<const char*>(buffer), mess_size};
  }
}

ReturnValue ReturnedCppException::pack() const
{
  const auto buffer_size = legion_buffer_size();
  const auto mem_kind    = find_memory_kind_for_executing_processor();
  auto buffer            = Legion::UntypedDeferredValue{buffer_size, mem_kind};
  const auto acc         = AccessorWO<std::int8_t, 1>{buffer, buffer_size, false};

  legion_serialize(acc.ptr(0));
  return {std::move(buffer), buffer_size};
}

std::string ReturnedCppException::to_string() const
{
  std::stringstream ss;

  ss << "ReturnedCppException(index = " << index_ << "message = " << message_ << ')';
  return std::move(ss).str();
}

}  // namespace legate::detail
