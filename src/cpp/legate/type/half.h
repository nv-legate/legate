/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#ifdef __CUDA_NO_HALF_CONVERSIONS__
#error "Built with CUDA but CUDA __half does not define conversion operators"
#endif

#ifdef __CUDA_NO_HALF_OPERATORS__
#error "Built with CUDA but CUDA __half does not define arithmetic operators"
#endif

#include <cuda/std/cmath>

#include <cuda_fp16.h>

#ifndef __CUDA_FP16_TYPES_EXIST__
#error "Built with CUDA but CUDA __half not defined"
#endif
#endif

#include <cstddef>
#include <cstdint>

/** @file */

namespace legate {

/**
 * @addtogroup types
 * @{
 */

#if LEGATE_DEFINED(LEGATE_USE_CUDA) && defined(CUDART_MAX_NORMAL_FP16)
#define LEGATE_MAX_HALF CUDART_MAX_NORMAL_FP16
#else
#define LEGATE_MAX_HALF 6.5504e+4F  // value of FLT16_MAX, which isn't always defined
#endif
#define LEGATE_MIN_HALF (-LEGATE_MAX_HALF)

#if LEGATE_DEFINED(LEGATE_DOXYGEN)
/**
 * @brief The half-precision floating point type used in Legate.
 *
 * Due to the incomplete support for half-precision floating point types in both C++ and
 * standard libraries, it is unspecified which operators and member functions exist on the
 * `Half` type, except:
 *
 * #. The type shall be constructible from and convertible to, `float`.
 * #. The type shall be assignable from `float`, i.e. `some_half = 1.0F` shall do what you
 *    expect.
 * #. When Legate is compiled with CUDA enabled, `Half` will be an alias to CUDA's native
 *    `__half` type.
 *
 * See
 * [`__half`](https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/group__CUDA__MATH__INTRINSIC__HALF.html)
 * for further information on the CUDA `__half` type.
 */
class Half {
 public:
  /**
   * @brief Construct a `Half` from a `float`.
   */
  explicit Half(float) noexcept;

  /**
   * @brief Assign a `float` to a `Half`.
   *
   * @return A reference to `this`.
   */
  Half& operator=(float) noexcept;

  /**
   * @brief Convert a `Half` to `float`.
   *
   * @return The current value as a `float`.
   */
  operator float() const noexcept;
};

#define LEGATE_DEFINED_HALF 1

#elif LEGATE_DEFINED(LEGATE_USE_CUDA)
using Half = ::__half;

#define LEGATE_DEFINED_HALF 0
#else

/**
 * @brief The half-precision floating point type used in Legate.
 *
 * This class exists if the system compiling Legate does not have native half support. It is
 * documented purely for development purposes. Other than the properties listed above in the
 * exposition-only documentation of Half, it offers no other guarantees or behaviors.
 */
class LEGATE_EXPORT Half {
 public:
  /**
   * @brief Default-constructs a Half with value 0.
   */
  constexpr Half() = default;

  /**
   * @brief Constructs a Half directly from its raw 16-bit representation.
   *
   * @param a The 16-bit encoded half-precision value.
   */
  constexpr explicit Half(std::uint16_t a) noexcept;

  /**
   * @brief Constructs a Half from a float value.
   *
   * @param a Single-precision value.
   */
  explicit Half(float a) noexcept;

  /**
   * @brief Constructs a Half from a double value.
   *
   * @param a Double-precision value.
   */
  explicit Half(double a) noexcept;

  /**
   * @brief Constructs a Half from an integer value.
   *
   * @param a Integer value.
   */
  explicit Half(int a) noexcept;

  /**
   * @brief Constructs a Half from a coord_t value.
   *
   * @param a coord_t value.
   */
  explicit Half(coord_t a) noexcept;

  /**
   * @brief Constructs a Half from a std::size_t value.
   *
   * @param a Size value.
   */
  explicit Half(std::size_t a) noexcept;

  /**
   * @brief Assigns a float value to this Half.
   */
  Half& operator=(float rhs) noexcept;

  /**
   * @brief Assigns a double value to this Half.
   */
  Half& operator=(double rhs) noexcept;

  /**
   * @brief Assigns an int value to this Half.
   */
  Half& operator=(int rhs) noexcept;

  /**
   * @brief Assigns a coord_t value to this Half.
   */
  Half& operator=(coord_t rhs) noexcept;

  /**
   * @brief Assigns a std::size_t value to this Half.
   */
  Half& operator=(std::size_t rhs) noexcept;

  /**
   * @brief Converts this Half to a float.
   *
   * @return The float representation.
   */
  [[nodiscard]] operator float() const noexcept;  // NOLINT(google-explicit-constructor)

  /**
   * @brief Converts this Half to a double.
   *
   * @return The double representation.
   */
  [[nodiscard]] explicit operator double() const noexcept;

  /**
   * @brief Adds another Half to this one.
   */
  Half& operator+=(const Half& rhs) noexcept;

  /**
   * @brief Subtracts another Half from this one.
   */
  Half& operator-=(const Half& rhs) noexcept;

  /**
   * @brief Multiplies this Half by another.
   */
  Half& operator*=(const Half& rhs) noexcept;

  /**
   * @brief Divides this Half by another.
   */
  Half& operator/=(const Half& rhs) noexcept;

  /**
   * @brief Returns the raw 16-bit representation of the Half.
   *
   * @return Encoded 16-bit value.
   */
  [[nodiscard]] constexpr std::uint16_t raw() const noexcept;

 private:
  std::uint16_t repr_{};
};

#define LEGATE_DEFINED_HALF 1

/**
 * @brief Unary negation.
 *
 * @param a Operand.
 *
 * @return Negated Half value.
 */
[[nodiscard]] LEGATE_EXPORT Half operator-(const Half& a) noexcept;

/**
 * @brief Addition of two Half values.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return Sum of a and b.
 */
[[nodiscard]] LEGATE_EXPORT Half operator+(const Half& a, const Half& b) noexcept;

/**
 * @brief Subtraction of two Half values.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return Difference a - b.
 */
[[nodiscard]] LEGATE_EXPORT Half operator-(const Half& a, const Half& b) noexcept;

/**
 * @brief Multiplication of two Half values.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return Product of a and b.
 */
[[nodiscard]] LEGATE_EXPORT Half operator*(const Half& a, const Half& b) noexcept;

/**
 * @brief Division of two Half values.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return Quotient a / b.
 */
[[nodiscard]] LEGATE_EXPORT Half operator/(const Half& a, const Half& b) noexcept;

/**
 * @brief Equality comparison.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return True if a and b are equal.
 */
[[nodiscard]] LEGATE_EXPORT bool operator==(const Half& a, const Half& b) noexcept;

/**
 * @brief Inequality comparison.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return True if a and b are not equal.
 */
[[nodiscard]] LEGATE_EXPORT bool operator!=(const Half& a, const Half& b) noexcept;

/**
 * @brief Less-than comparison.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return True if a < b.
 */
[[nodiscard]] LEGATE_EXPORT bool operator<(const Half& a, const Half& b) noexcept;

/**
 * @brief Less-than-or-equal comparison.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return True if a <= b.
 */
[[nodiscard]] LEGATE_EXPORT bool operator<=(const Half& a, const Half& b) noexcept;

/**
 * @brief Greater-than comparison.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return True if a > b.
 */
[[nodiscard]] LEGATE_EXPORT bool operator>(const Half& a, const Half& b) noexcept;

/**
 * @brief Greater-than-or-equal comparison.
 *
 * @param a Left operand.
 * @param b Right operand.
 *
 * @return True if a >= b.
 */
[[nodiscard]] LEGATE_EXPORT bool operator>=(const Half& a, const Half& b) noexcept;
#endif

/** @} */

}  // namespace legate

#include <legate/type/half.inl>
