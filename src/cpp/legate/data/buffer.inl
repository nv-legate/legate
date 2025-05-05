/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/buffer.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/machine.h>

#include <cstddef>
#include <cstdint>

namespace legate {

namespace untyped_buffer_detail {

void check_type(const Type& ty, std::size_t size_of, std::size_t align_of);

}  // namespace untyped_buffer_detail

template <typename T, std::int32_t DIM>
TaskLocalBuffer::TaskLocalBuffer(const Buffer<T, DIM>& buf, const Type& type)
  : TaskLocalBuffer{buf, type, buf.get_bounds()}
{
  untyped_buffer_detail::check_type(type, sizeof(T), alignof(T));
}

template <typename T, std::int32_t DIM>
TaskLocalBuffer::TaskLocalBuffer(const Buffer<T, DIM>& buf)
  : TaskLocalBuffer{buf, primitive_type(type_code_of_v<T>)}
{
  static_assert(type_code_of_v<T> != Type::Code::FIXED_ARRAY);
  static_assert(type_code_of_v<T> != Type::Code::STRUCT);
  static_assert(type_code_of_v<T> != Type::Code::STRING);
  static_assert(type_code_of_v<T> != Type::Code::NIL);
}

template <typename T, std::int32_t DIM>
TaskLocalBuffer::operator Buffer<T, DIM>() const
{
  return static_cast<Buffer<T, DIM>>(legion_buffer_());
}

inline const SharedPtr<detail::TaskLocalBuffer>& TaskLocalBuffer::impl() const { return impl_; }

// ==========================================================================================

namespace detail {

void check_alignment(std::size_t alignment);

}  // namespace detail

template <typename VAL, std::int32_t DIM>
Buffer<VAL, DIM> create_buffer(const Point<DIM>& extents, Memory::Kind kind, std::size_t alignment)
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  detail::check_alignment(alignment);

  if (Memory::Kind::NO_MEMKIND == kind) {
    kind = find_memory_kind_for_executing_processor(false);
  }
  auto hi = extents - Point<DIM>::ONES();
  return Buffer<VAL, DIM>{Rect<DIM>{Point<DIM>::ZEROES(), std::move(hi)}, kind, nullptr, alignment};
}

template <typename VAL>
Buffer<VAL> create_buffer(std::size_t size, Memory::Kind kind, std::size_t alignment)
{
  return create_buffer<VAL, 1>(Point<1>{size}, kind, alignment);
}

}  // namespace legate
