/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/mdspan/flat_mdspan_view.h>

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
constexpr FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::FlatMDSpanView(
  mdspan_type span) noexcept
  : span_{std::move(span)}
{
}

template <typename El, typename Ex, typename L, typename A>
constexpr typename FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::iterator
FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::begin() const noexcept
{
  return cbegin();
}

template <typename El, typename Ex, typename L, typename A>
constexpr typename FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::iterator
FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::cbegin() const noexcept
{
  return iterator{{}, span_, 0};
}

template <typename El, typename Ex, typename L, typename A>
constexpr typename FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::iterator
FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::end() const noexcept
{
  return cend();
}

template <typename El, typename Ex, typename L, typename A>
constexpr typename FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::iterator
FlatMDSpanView<::cuda::std::mdspan<El, Ex, L, A>>::cend() const noexcept
{
  return iterator{{}, span_, static_cast<typename mdspan_type::index_type>(span_.size())};
}

#endif

}  // namespace legate::detail
