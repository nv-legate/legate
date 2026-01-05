/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/type/complex.h>
#include <legate/type/half.h>

namespace legate::detail {

/**
 * @brief The atomic wrapper for Complex<Half>
 *
 * @tparam F The non-atomic binary functor.
 *
 * libcu++ does not yet have suitable overloads for cuda::std::atomic_ref for Complex<Half>,
 * and so we must roll our own.
 */
template <typename F>
class AtomicWrapperComplexHalf {
 public:
  /**
   * @brief Atomically applies the functor to lhs with rhs.
   *
   * @param lhs Reference to the Complex<Half> value to update.
   * @param rhs Value to apply.
   */
  LEGATE_HOST_DEVICE void operator()(Complex<Half>& lhs, Complex<Half> rhs) const noexcept;

 private:
  /**
   * @brief Host implementation of the atomic apply.
   */
  LEGATE_HOST static void apply_host_(Complex<Half>& lhs, Complex<Half> rhs) noexcept;

  /**
   * @brief Device implementation of the atomic apply.
   */
  LEGATE_DEVICE static void apply_device_(Complex<Half>& lhs, Complex<Half> rhs) noexcept;
};

// ==========================================================================================

/**
 * @brief The atomic wrapper for Complex<Half>
 *
 * @tparam F The non-atomic binary functor.
 *
 * libcu++ does not yet have suitable overloads for cuda::std::atomic_ref for Complex<double>,
 * because cuda::std::atomic_ref requires that the type is between 8 and 64 bits
 * wide. Complex<double> is 128 bits wide.
 */
template <typename F>
class AtomicWrapperComplexDouble {
 public:
  /**
   * @brief Atomically applies the functor to lhs with rhs.
   *
   * @param lhs Reference to the Complex<double> value to update.
   * @param rhs Value to apply.
   */
  LEGATE_HOST_DEVICE void operator()(Complex<double>& lhs, Complex<double> rhs) const noexcept;

 private:
  /**
   * @brief Host implementation of the atomic apply.
   */
  LEGATE_HOST static void apply_host_(Complex<double>& lhs, Complex<double> rhs) noexcept;

  /**
   * @brief Device implementation of the atomic apply.
   */
  LEGATE_DEVICE static void apply_device_(Complex<double>& lhs, Complex<double> rhs) noexcept;
};

}  // namespace legate::detail

#include <legate/redop/detail/complex.inl>
