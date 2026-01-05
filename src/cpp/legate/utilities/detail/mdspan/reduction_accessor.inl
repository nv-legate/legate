/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/mdspan/reduction_accessor.h>

#include <cuda/std/atomic>

namespace legate::detail {

template <typename R, bool E>
constexpr ReductionAccessor<R, E>::ReferenceWrapper::ReferenceWrapper(
  data_handle_type elem) noexcept
  : elem_{elem}
{
}

template <typename R, bool E>
template <typename T>
constexpr typename ReductionAccessor<R, E>::ReferenceWrapper&
ReductionAccessor<R, E>::ReferenceWrapper::operator=(const T&) noexcept
{
  static_assert(sizeof(T*) != sizeof(T*),  // NOLINT(misc-redundant-expression)
                "Reduction accessors may not be written to directly, you may only reduce into them "
                "(via reduce()) or atomically read the current value (via get())");
}

template <typename R, bool E>
constexpr void ReductionAccessor<R, E>::ReferenceWrapper::reduce(
  const typename reduction_type::RHS& val) noexcept
{
  reduction_type::template fold<EXCLUSIVE>(*data(), val);
}

template <typename R, bool E>
constexpr void ReductionAccessor<R, E>::ReferenceWrapper::operator<<=(
  const typename reduction_type::RHS& val) noexcept
{
  reduce(val);
}

template <typename R, bool E>
constexpr std::conditional_t<E,
                             const typename ReductionAccessor<R, E>::element_type&,
                             typename ReductionAccessor<R, E>::element_type>
ReductionAccessor<R, E>::ReferenceWrapper::get() const noexcept
{
  if constexpr (EXCLUSIVE) {
    return *data();
  } else {
    return ::cuda::std::atomic_ref<element_type>{*data()}.load();
  }
}

template <typename R, bool E>
constexpr typename ReductionAccessor<R, E>::data_handle_type
ReductionAccessor<R, E>::ReferenceWrapper::data() const noexcept
{
  return elem_;
}

// ------------------------------------------------------------------------------------------

template <typename R, bool E>
template <typename U, bool UExcl, typename SFINAE>
constexpr ReductionAccessor<R, E>::ReductionAccessor(const ReductionAccessor<U, UExcl>&) noexcept
{
}

template <typename R, bool E>
constexpr typename ReductionAccessor<R, E>::reference ReductionAccessor<R, E>::access(
  data_handle_type p, std::size_t i) const noexcept
{
  return reference{offset(p, i)};
}

template <typename R, bool E>
constexpr typename ReductionAccessor<R, E>::data_handle_type ReductionAccessor<R, E>::offset(
  data_handle_type p, std::size_t i) const noexcept
{
  return p + i;
}

}  // namespace legate::detail
