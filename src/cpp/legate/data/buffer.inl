/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/buffer.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/machine.h>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace legate {

namespace detail {

LEGATE_EXPORT void check_alignment(std::size_t alignment);

}  // namespace detail

template <typename VAL, std::int32_t DIM>
Buffer<VAL, DIM>::Buffer(InternalSharedPtr<detail::InlineStorage> storage, const Rect<DIM>& bounds)
  : storage_{std::move(storage)}, bounds_{bounds}
{
  void* base                                 = nullptr;
  std::uint64_t byte_strides[LEGATE_MAX_DIM] = {};

  detail::inline_storage_layout(storage_, &base, byte_strides);
  base_ = static_cast<VAL*>(base);
  for (std::int32_t i = 0; i < DIM; ++i) {
    strides_[i] = static_cast<std::size_t>(byte_strides[i]) / sizeof(VAL);
  }
}

namespace untyped_buffer_detail {

LEGATE_EXPORT void check_type(const Type& ty, std::size_t size_of, std::size_t align_of);

}  // namespace untyped_buffer_detail

template <typename T, std::int32_t DIM>
TaskLocalBuffer::TaskLocalBuffer(const Buffer<T, DIM>& buf, const Type& type)
  : TaskLocalBuffer{buf.get_storage(), type, Domain{buf.get_bounds()}}
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
  return Buffer<T, DIM>{storage_(), domain().bounds<DIM, coord_t>()};
}

inline const SharedPtr<detail::TaskLocalBuffer>& TaskLocalBuffer::impl() const { return impl_; }

// ==========================================================================================

template <typename VAL, std::int32_t DIM>
Buffer<VAL, DIM> create_buffer(const Point<DIM>& extents, Memory::Kind kind, std::size_t alignment)
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  detail::check_alignment(alignment);

  if (Memory::Kind::NO_MEMKIND == kind) {
    kind = find_memory_kind_for_executing_processor(/* host_accessible */ false);
  }

  const auto bounds = Rect<DIM>{Point<DIM>::ZEROES(), extents - Point<DIM>::ONES()};

  return Buffer<VAL, DIM>{detail::create_buffer_storage(sizeof(VAL), Domain{bounds}, kind), bounds};
}

template <typename VAL, std::int32_t DIM>
Buffer<VAL, DIM> create_buffer(const Point<DIM>& extents, Memory mem, std::size_t alignment)
{
  return create_buffer<VAL, DIM>(extents, mem.kind(), alignment);
}

template <typename VAL>
Buffer<VAL> create_buffer(std::size_t size, Memory::Kind kind, std::size_t alignment)
{
  return create_buffer<VAL, 1>(Point<1>{size}, kind, alignment);
}

template <typename VAL>
Buffer<VAL> create_buffer(std::size_t size, Memory mem, std::size_t alignment)
{
  return create_buffer<VAL, 1>(Point<1>{size}, mem, alignment);
}

}  // namespace legate
