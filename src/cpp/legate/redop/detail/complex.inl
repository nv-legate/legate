/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/redop/detail/atomic_wrapper.h>
#include <legate/redop/detail/complex.h>
#include <legate/utilities/macros.h>

#include <cuda/std/atomic>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace legate::detail {

template <typename F>
LEGATE_HOST /*static*/ void AtomicWrapperComplexHalf<F>::apply_host_(Complex<Half>& lhs,
                                                                     Complex<Half> rhs) noexcept
{
  // Currently __half2 is not considered trivially copyable, because it defines a copy
  // constructor, and hence cuda::std::atomic_ref complains. If that ever changes, we should
  // ditch the type punning business below and just use the standard stuff.
  if constexpr (!LEGATE_DEFINED(LEGATE_DEFINED_HALF)) {
    static_assert(
      !std::is_trivially_copyable_v<Complex<Half>>,
      "__half2 is trivially copyable now, can use atomic_ref with cuda::std::complex now!");
  }
  using storage_type = std::int32_t;

  constexpr auto bitcast_int32_t_as_complex_half = [](storage_type src) {
    Complex<Half> dest;

    // The static_cast is required to silence the compiler:
    //
    // src/cpp/legate/redop/detail/complex.inl:45:16: error: 'void* memcpy(void*, const void*,
    // std::size_t)' writing to an object of type 'legate::Complex<__half>' {aka 'class
    // cuda::std::__4::complex<__half>'} with no trivial copy-assignment; use copy-assignment
    // or copy-initialization instead [-Werror=class-memaccess]
    //
    //    45 |     std::memcpy(&dest, &src, sizeof(src));
    //       |     ~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // We are deliberately type-punning the objects here in order to treat Complex<Half> as a
    // packed int32_t.
    static_assert(sizeof(dest) == sizeof(src));
    std::memcpy(static_cast<void*>(&dest), &src, sizeof(src));
    return dest;
  };

  constexpr auto bitcast_complex_half_as_int32_t = [](const Complex<Half>& src) {
    storage_type dest;

    static_assert(sizeof(dest) == sizeof(src));
    std::memcpy(&dest, static_cast<const void*>(&src), sizeof(src));
    return dest;
  };

  auto* const lhs_ptr = reinterpret_cast<storage_type*>(&lhs);
  auto lhs_ref        = ::cuda::std::atomic_ref<storage_type>{*lhs_ptr};
  auto old_val        = lhs_ref.load(::cuda::std::memory_order_relaxed);
  storage_type new_val;

  do {
    const auto old_tmp = bitcast_int32_t_as_complex_half(old_val);
    const auto tmp     = F{}(old_tmp, rhs);

    new_val = bitcast_complex_half_as_int32_t(tmp);
  } while (!lhs_ref.compare_exchange_weak(
    old_val, new_val, ::cuda::std::memory_order_relaxed, ::cuda::std::memory_order_relaxed));
}

template <typename F>
LEGATE_DEVICE /*static*/ void AtomicWrapperComplexHalf<F>::apply_device_(
  Complex<Half>& lhs,
  Complex<Half> rhs  // NOLINT(performance-unnecessary-value-param)
  ) noexcept
{
#if LEGATE_DEFINED(LEGATE_DEVICE_COMPILE)
  using storage_type = std::uint32_t;

  constexpr auto bitcast_complex_as_uint = [](const Complex<Half>& value) {
    // clang-tidy claims this call to real() and imag() are using uninitialized values. I
    // cannot for the life of me tell where these are not initialized.
    //
    // NOLINTBEGIN(clang-analyzer-core.uninitialized.UndefReturn)
    const auto real = __half_as_ushort(value.real());
    const auto imag = __half_as_ushort(value.imag());
    // NOLINTEND(clang-analyzer-core.uninitialized.UndefReturn)
    storage_type result;

    static_assert(sizeof(real) + sizeof(imag) == sizeof(result));
    asm("mov.b32 %0, {%1,%2};" : "=r"(result) : "h"(real), "h"(imag));
    return result;
  };

  constexpr auto bitcast_uint_as_complex = [](storage_type value) -> Complex<Half> {
    std::uint16_t real;
    std::uint16_t imag;

    asm("mov.b32 {%0,%1}, %2;" : "=h"(real), "=h"(imag) : "r"(value));
    return {__ushort_as_half(real), __ushort_as_half(imag)};
  };

  auto newval = lhs;
  Complex<Half> oldval;
  // Type punning like this is illegal in C++ but the CUDA manual has an example just like it
  auto* const ptr = reinterpret_cast<std::uint32_t*>(&lhs);

  do {
    oldval = newval;
    newval = F{}(newval, rhs);
    newval = bitcast_uint_as_complex(
      atomicCAS(ptr, bitcast_complex_as_uint(oldval), bitcast_complex_as_uint(newval)));
  } while (oldval != newval);
#else
  static_cast<void>(lhs);
  static_cast<void>(rhs);
#endif
}

// ------------------------------------------------------------------------------------------

template <typename F>
LEGATE_HOST_DEVICE void AtomicWrapperComplexHalf<F>::operator()(Complex<Half>& lhs,
                                                                Complex<Half> rhs) const noexcept
{
  if constexpr (LEGATE_DEFINED(LEGATE_DEVICE_COMPILE)) {
    apply_device_(lhs, std::move(rhs));
  } else {
    apply_host_(lhs, std::move(rhs));
  }
}

// ==========================================================================================

template <typename F>
LEGATE_HOST /*static*/ void AtomicWrapperComplexDouble<F>::apply_host_(Complex<double>& lhs,
                                                                       Complex<double> rhs) noexcept
{
  AtomicWrapper<F>{}(lhs, rhs);
}

template <typename F>
LEGATE_DEVICE /*static*/ void AtomicWrapperComplexDouble<F>::apply_device_(
  Complex<double>& lhs, Complex<double> rhs) noexcept
{
#if LEGATE_DEFINED(LEGATE_DEVICE_COMPILE)
  constexpr auto bitcast_double_as_uint64_t = [](double value) {
    std::uint64_t result;

    static_assert(sizeof(value) == sizeof(result));
    static_assert(sizeof(value) == 8);
    asm("mov.b64 %0, %1;" : "=l"(result) : "d"(value));
    return result;
  };

  constexpr auto bitcast_uint64_t_as_double = [](std::uint64_t value) {
    double result;

    static_assert(sizeof(value) == sizeof(result));
    static_assert(sizeof(value) == 8);
    asm("mov.b64 %0, %1;" : "=d"(result) : "l"(value));
    return result;
  };

  double oldval;
  static_assert(sizeof(lhs) == 2 * sizeof(std::uint64_t),
                "Type punning of Complex<double> as 2 packed uint64_t's will not work!");
  // Type punning like this is illegal in C++ but the CUDA manual has an example just like it
  // Silence clang-tidy (which wants us to use std::uint64_t) because std::uint64_t is not
  // guaranteed to be defined as unsigned long long, which is what atomicCAS() expects.
  auto* const ptr = reinterpret_cast<unsigned long long*>(&lhs);  // NOLINT(google-runtime-int)
  auto newval     = lhs.real();

  do {
    oldval = newval;
    newval = F{}(newval, rhs.real());
    newval = bitcast_uint64_t_as_double(
      atomicCAS(ptr, bitcast_double_as_uint64_t(oldval), bitcast_double_as_uint64_t(newval)));
  } while (oldval != newval);
  newval = lhs.imag();
  do {
    oldval = newval;
    newval = F{}(newval, rhs.imag());
    newval = bitcast_uint64_t_as_double(
      atomicCAS(ptr + 1, bitcast_double_as_uint64_t(oldval), bitcast_double_as_uint64_t(newval)));
  } while (oldval != newval);
#else
  static_cast<void>(lhs);
  static_cast<void>(rhs);
#endif
}

// ------------------------------------------------------------------------------------------

template <typename F>
LEGATE_HOST_DEVICE void AtomicWrapperComplexDouble<F>::operator()(
  Complex<double>& lhs, Complex<double> rhs) const noexcept
{
  if constexpr (LEGATE_DEFINED(LEGATE_DEVICE_COMPILE)) {
    apply_device_(lhs, std::move(rhs));
  } else {
    apply_host_(lhs, std::move(rhs));
  }
}

}  // namespace legate::detail
