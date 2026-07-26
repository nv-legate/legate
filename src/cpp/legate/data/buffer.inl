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

LEGATE_EXPORT void check_type(const legate::Type& ty, std::size_t size_of, std::size_t align_of);

LEGATE_EXPORT void check_alignment(std::size_t alignment);

void throw_out_of_bounds(const Rect& domain, const Rect& to_access);

// Synthesizes the element `Type` for a `create_buffer<VAL>()` allocation: a primitive type when
// `VAL` maps to one, otherwise an opaque binary type of the right size.
template <typename VAL>
[[nodiscard]] inline legate::Type type_of()
{
  constexpr auto CODE = type_code_of_v<VAL>;

  if constexpr (CODE == legate::Type::Code::FIXED_ARRAY || CODE == legate::Type::Code::STRUCT) {
    return legate::binary_type(sizeof(VAL));
  } else {
    return legate::primitive_type(CODE);
  }
}

}  // namespace detail

// ==========================================================================================

template <typename VAL, std::int32_t DIM>
LEGATE_HOST_DEVICE VAL& NDBuffer<VAL, DIM>::operator[](const NDPoint<DIM>& p) const
{
  return *ptr(p);
}

template <typename VAL, std::int32_t DIM>
LEGATE_HOST_DEVICE VAL* NDBuffer<VAL, DIM>::ptr(const NDPoint<DIM>& p) const
{
  return base_ + p.dot(strides_);
}

template <typename VAL, std::int32_t DIM>
LEGATE_HOST_DEVICE VAL* NDBuffer<VAL, DIM>::ptr(const NDRect<DIM>& r,
                                                std::size_t strides[DIM]) const
{
  if (!domain_.contains(r)) {
    detail::throw_out_of_bounds(Rect{domain_}, Rect{r});
  }
  for (std::int32_t i = 0; i < DIM; ++i) {
    strides[i] = static_cast<std::size_t>(strides_[i]);
  }
  return ptr(r.lo);
}

template <typename VAL, std::int32_t DIM>
LEGATE_HOST_DEVICE NDRect<DIM> NDBuffer<VAL, DIM>::domain() const
{
  return domain_;
}

template <typename VAL, std::int32_t DIM>
const Buffer& NDBuffer<VAL, DIM>::buffer() const
{
  return buffer_;
}

template <typename VAL, std::int32_t DIM>
void NDBuffer<VAL, DIM>::destroy()
{
  buffer_ = Buffer{};
  base_   = nullptr;
}

template <typename VAL, std::int32_t DIM>
NDBuffer<VAL, DIM>::NDBuffer(Buffer buffer) : buffer_{std::move(buffer)}
{
  detail::check_type(buffer_.type(), sizeof(VAL), alignof(VAL));
  init_();
}

template <typename VAL, std::int32_t DIM>
void NDBuffer<VAL, DIM>::init_()
{
  auto&& [base, strides] = buffer().base_and_element_strides();

  base_    = static_cast<VAL*>(base);
  strides_ = static_cast<NDPoint<DIM>>(strides);
  domain_  = static_cast<NDRect<DIM>>(buffer().domain());
}

template <typename VAL, std::int32_t DIM>
NDBuffer<VAL, DIM>::NDBuffer(Buffer buffer, InitTag) : buffer_{std::move(buffer)}
{
  init_();
}

// ==========================================================================================

template <typename VAL, std::int32_t DIM>
NDBuffer<VAL, DIM> create_buffer(const NDPoint<DIM>& extents,
                                 Memory::Kind kind,
                                 std::size_t alignment)
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  detail::check_alignment(alignment);

  if (Memory::Kind::NO_MEMKIND == kind) {
    kind = find_memory_kind_for_executing_processor(/* host_accessible */ false);
  }

  const auto bounds = NDRect<DIM>{NDPoint<DIM>::zeroes(), extents - NDPoint<DIM>::ones()};
  auto buffer       = Buffer::allocate(detail::type_of<VAL>(), Rect{bounds}, kind);

  return NDBuffer<VAL, DIM>{std::move(buffer), typename NDBuffer<VAL, DIM>::InitTag{}};
}

template <typename VAL, std::int32_t DIM>
NDBuffer<VAL, DIM> create_buffer(const NDPoint<DIM>& extents, Memory mem, std::size_t alignment)
{
  return create_buffer<VAL, DIM>(extents, mem.kind(), alignment);
}

template <typename VAL>
NDBuffer<VAL> create_buffer(std::size_t size, Memory::Kind kind, std::size_t alignment)
{
  return create_buffer<VAL, 1>(NDPoint<1>{static_cast<std::int64_t>(size)}, kind, alignment);
}

template <typename VAL>
NDBuffer<VAL> create_buffer(std::size_t size, Memory mem, std::size_t alignment)
{
  return create_buffer<VAL, 1>(NDPoint<1>{static_cast<std::int64_t>(size)}, mem, alignment);
}

inline const SharedPtr<detail::Buffer>& Buffer::impl() const { return impl_; }

}  // namespace legate
