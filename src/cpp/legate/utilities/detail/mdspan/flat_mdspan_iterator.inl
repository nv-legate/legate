/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/assert.h>
#include <legate/utilities/detail/mdspan/flat_mdspan_iterator.h>

#include <cuda/std/array>

#include <cstddef>
#include <memory>
#include <type_traits>

namespace legate::detail {

// Doxygen seemingly does not understand template class partial specialization. I.e. it looks
// for FlatMDSpanIterator<T> (because that's what the underlying type is declared as) and does
// not understand that
//
// template <typename T, typename U>
// class Foo<OtherType<T, U>>
//
// is *also* a Foo<T>, and is a unique type of it. So we get a bunch of
//
// src/cpp/legate/utilities/mdspan.inl:15: error: no uniquely matching class member found for
//   template < El, Ex, L, A >
//   constexpr El * legate::detail::FlatMDSpanIterator::pointer_type::operator->() const
// Possible candidates:
// ...
// 'constexpr Element * legate::detail::FlatMDSpanIterator<::cuda::std::mdspan< Element,
// Extent, Layout, Accessor > >::pointer_type::operator->() const noexcept' at line 88 of file
// /Users/jfaibussowit/soft/nv/legate.core.internal/src/cpp/legate/utilities/mdspan.h
// ...
#ifndef DOXYGEN

template <typename El, typename Ex, typename L, typename A>
constexpr El* FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::PointerWrapper::operator->()
  const noexcept
{
  return std::addressof(elem_);
}

// ------------------------------------------------------------------------------------------

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::FlatMDSpanIterator(
  ConstructKey, const mdspan_type& span, index_type idx) noexcept
  : span_{std::addressof(span)}, idx_{idx}
{
}

template <typename El, typename Ex, typename L, typename A>
constexpr typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::reference
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator*() const noexcept
{
  constexpr auto DIM = mdspan_type::rank();

  if constexpr (DIM <= 1) {
    // The std::array implementation is correct for any dimension, but going through it appears
    // to confuse the optimizers and they generate very suboptimal code...
    //
    // So for 1D access we can skip it (there is no need to transform our offset) and can use
    // our `idx_` directly.
    return (*span_)(idx_);
  } else {
    // This code right here is why flat mdspan iterators are (currently) terribly
    // inefficient. The compilers will able to optimize this loop to its minimum, but they are
    // *not* able to optimize the outer loop to get rid of this calculation. For some reason,
    // this code below is complex enough to defeat the vectorization engines of every major
    // compiler.
    //
    // Concretely, a completely equivalent loop:
    //
    // for (std::size_t i = 0; i < span.extent(0); ++i) {
    //   for (std::size_t j = 0; j < span.extent(1); ++j) {
    //     span(i, j) = ...
    //   }
    // }
    //
    // Will be fully vectorized by optimizers, but the following (which is more or less what
    // our iterator expands to):
    //
    // for (std::size_t i = 0; i < PROD(span.extents()...); ++i) {
    //   std::array<std::size_t, DIM> md_idx = delinearize(i); // this is us
    //
    //   span(md_idx) = ...
    // }
    //
    // Is considered too complicated to unravel. A shame. Hopefully optimizers eventually
    // become smart enough to unravel this.
    ::cuda::std::array<index_type, DIM> md_idx;
    {
      auto index = idx_;  // Cannot declare in for-loop, since index_type != rank_type

      for (auto dim = DIM; dim-- > 0;) {
        md_idx[dim] = index % span_->extent(dim);
        index /= span_->extent(dim);
      }
    }
    return (*span_)[md_idx];
  }
}

template <typename El, typename Ex, typename L, typename A>
constexpr typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::pointer
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator->() const noexcept
{
  if constexpr (std::is_lvalue_reference_v<reference>) {
    return &operator*();
  } else {
    return PointerWrapper{operator*()};
  }
}

namespace cmp_detail {

template <typename T, typename U>
constexpr bool cmp_less(T t, U u) noexcept
{
  LEGATE_CPP_VERSION_TODO(20, "Use std::cmp_less() instead");
  if constexpr (std::is_signed_v<T> == std::is_signed_v<U>) {
    return t < u;
  } else if constexpr (std::is_signed_v<T>) {
    return t < 0 || std::make_unsigned_t<T>(t) < u;
  } else {
    return u >= 0 && t < std::make_unsigned_t<U>(u);
  }
}

template <typename T, typename U>
constexpr bool cmp_greater_equal(T t, U u) noexcept
{
  return !cmp_less(t, u);
}

template <typename T, typename U>
constexpr bool cmp_less_equal(T t, U u) noexcept
{
  return !cmp_less(u, t);
}

}  // namespace cmp_detail

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>&
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator++() noexcept
{
  LEGATE_ASSERT(cmp_detail::cmp_less(idx_, span_->size()));
  ++idx_;
  return *this;
}

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator++(int) noexcept
{
  auto copy = *this;

  ++(*this);
  return copy;
}

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>&
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator--() noexcept
{
  LEGATE_ASSERT(idx_ > 0);
  --idx_;
  return *this;
}

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator--(int) noexcept
{
  auto copy = *this;

  --(*this);
  return copy;
}

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>&
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator+=(difference_type n) noexcept
{
  if (n < 0) {
    LEGATE_ASSERT(cmp_detail::cmp_greater_equal(idx_, -n));
    idx_ -= static_cast<index_type>(-n);
  } else {
    LEGATE_ASSERT(cmp_detail::cmp_less_equal(idx_ + static_cast<index_type>(n), span_->size()));
    idx_ += static_cast<index_type>(n);
  }
  return *this;
}

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>&
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator-=(difference_type n) noexcept
{
  return operator+=(-n);
}

template <typename El, typename Ex, typename L, typename A>
constexpr typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::reference
FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::operator[](difference_type n) const noexcept
{
  return *(*this + n);
}

// ==========================================================================================

template <typename El, typename Ex, typename L, typename A>
constexpr typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type operator-(
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& self,
  const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& other) noexcept
{
  using difference_type =
    typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type;

  LEGATE_ASSERT(self.span_ == other.span_);
  return static_cast<difference_type>(self.idx_) - static_cast<difference_type>(other.idx_);
}

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> operator+(
  FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> self,
  typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type n) noexcept
{
  self += n;
  return self;
}

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> operator+(
  typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type n,
  FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> self) noexcept
{
  self += n;
  return self;
}

template <typename El, typename Ex, typename L, typename A>
constexpr FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> operator-(
  FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>> self,
  typename FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>::difference_type n) noexcept
{
  self -= n;
  return self;
}

template <typename El, typename Ex, typename L, typename A>
constexpr bool operator==(const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
                          const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept
{
  LEGATE_ASSERT(lhs.span_ == rhs.span_);
  return lhs.idx_ == rhs.idx_;
}

template <typename El, typename Ex, typename L, typename A>
constexpr bool operator!=(const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
                          const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept
{
  return !(lhs == rhs);
}

template <typename El, typename Ex, typename L, typename A>
constexpr bool operator<(const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
                         const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept
{
  LEGATE_ASSERT(lhs.span_ == rhs.span_);
  return lhs.idx_ < rhs.idx_;
}

template <typename El, typename Ex, typename L, typename A>
constexpr bool operator>(const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
                         const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept
{
  return rhs < lhs;
}

template <typename El, typename Ex, typename L, typename A>
constexpr bool operator<=(const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
                          const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept
{
  return !(rhs < lhs);
}

template <typename El, typename Ex, typename L, typename A>
constexpr bool operator>=(const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& lhs,
                          const FlatMDSpanIterator<::cuda::std::mdspan<El, Ex, L, A>>& rhs) noexcept
{
  return !(lhs < rhs);
}

#endif

}  // namespace legate::detail
