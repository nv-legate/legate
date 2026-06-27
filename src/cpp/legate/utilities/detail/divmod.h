/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <cstdint>
#include <stdlib.h>

/// Optionally enable GCC's built-in type
#if !defined(__CUDA_ARCH__) && defined(__GNUC__) && defined(__SIZEOF_INT128__)
#define UINT128_NATIVE
#endif

#define __CUDA_HD__ LEGATE_HOST LEGATE_DEVICE

namespace legate::detail {

// uint128_t for host and device from CUTLASS

__extension__ typedef unsigned __int128 uint128;

///! Unsigned 128b integer type
struct uint128_t {
  /// Size of one part of the uint's storage in bits
  static constexpr int kPartSize = 8 * sizeof(std::uint64_t);

  struct hilo {
    std::uint64_t lo;
    std::uint64_t hi;

    __CUDA_HD__ hilo(std::uint64_t lo_, std::uint64_t hi_) : lo(lo_), hi(hi_) {}
  };

  // Use a union to store either low and high parts or, if present, a built-in
  // 128b integer type.
  union {
    struct hilo hilo_;

#if defined(UINT128_NATIVE)
    uint128 native;
#endif  // defined(UINT128_NATIVE)
  };

  //
  // Methods
  //

  /// Default ctor
  __CUDA_HD__
  uint128_t() : hilo_(0, 0) {}

  /// Constructor from uint64
  __CUDA_HD__
  uint128_t(std::uint64_t lo_) : hilo_(lo_, 0) {}

  /// Constructor from two 64b unsigned integers
  __CUDA_HD__
  uint128_t(std::uint64_t lo_, std::uint64_t hi_) : hilo_(lo_, hi_) {}

/// Optional constructor from native value
#if defined(UINT128_NATIVE)
  uint128_t(uint128 value) : native(value) {}
#endif

  /// Lossily cast to uint64
  __CUDA_HD__
  explicit operator uint64_t() const { return hilo_.lo; }

  __CUDA_HD__
  static void exception()
  {
#if defined(__CUDA_ARCH__)
    asm volatile("  brkpt;\n");
#else
    // throw std::runtime_error("Not yet implemented.");
    std::abort();
#endif
  }

  /// Add
  __CUDA_HD__
  uint128_t operator+(uint128_t const& rhs) const
  {
    uint128_t y;
#if defined(UINT128_NATIVE)
    y.native = native + rhs.native;
#else
    y.hilo_.lo = hilo_.lo + rhs.hilo_.lo;
    y.hilo_.hi = hilo_.hi + rhs.hilo_.hi + (!y.hilo_.lo && (rhs.hilo_.lo));
#endif
    return y;
  }

  /// Subtract
  __CUDA_HD__
  uint128_t operator-(uint128_t const& rhs) const
  {
    uint128_t y;
#if defined(UINT128_NATIVE)
    y.native = native - rhs.native;
#else
    y.hilo_.lo = hilo_.lo - rhs.hilo_.lo;
    y.hilo_.hi = hilo_.hi - rhs.hilo_.hi - (rhs.hilo_.lo && y.hilo_.lo > hilo_.lo);
#endif
    return y;
  }

  /// Multiply by unsigned 64b integer yielding 128b integer
  __CUDA_HD__
  uint128_t operator*(std::uint64_t const& rhs) const
  {
    uint128_t y;
#if defined(UINT128_NATIVE)
    y.native = native * rhs;
#else
    // TODO - not implemented
    exception();
#endif
    return y;
  }

  /// Divide 128b operation by 64b operation yielding a 64b quotient
  __CUDA_HD__
  std::uint64_t operator/(std::uint64_t const& divisor) const
  {
    std::uint64_t quotient = 0;
#if defined(UINT128_NATIVE)
    quotient = uint64_t(native / divisor);
#else
    // TODO - not implemented
    exception();
#endif
    return quotient;
  }

  /// Divide 128b operation by 64b operation yielding a 64b quotient
  __CUDA_HD__
  std::uint64_t operator%(std::uint64_t const& divisor) const
  {
    std::uint64_t remainder = 0;
#if defined(UINT128_NATIVE)
    remainder = uint64_t(native % divisor);
#else
    // TODO - not implemented
    exception();
#endif
    return remainder;
  }

  /// Computes the quotient and remainder in a single method.
  __CUDA_HD__
  std::uint64_t divmod(std::uint64_t& remainder, std::uint64_t divisor) const
  {
    std::uint64_t quotient = 0;
#if defined(UINT128_NATIVE)
    quotient  = uint64_t(native / divisor);
    remainder = uint64_t(native % divisor);
#else
    // TODO - not implemented
    exception();
#endif
    return quotient;
  }

  /// Left-shifts a 128b unsigned integer
  __CUDA_HD__
  uint128_t operator<<(int sh) const
  {
    if (sh == 0) {
      return *this;
    } else if (sh >= kPartSize) {
      return uint128_t(0, hilo_.lo << (sh - kPartSize));
    } else {
      return uint128_t((hilo_.lo << sh), (hilo_.hi << sh) | uint64_t(hilo_.lo >> (kPartSize - sh)));
    }
  }

  /// Right-shifts a 128b unsigned integer
  __CUDA_HD__
  uint128_t operator>>(int sh) const
  {
    if (sh == 0) {
      return *this;
    } else if (sh >= kPartSize) {
      return uint128_t((hilo_.hi >> (sh - kPartSize)), 0);
    } else {
      return uint128_t((hilo_.lo >> sh) | (hilo_.hi << (kPartSize - sh)), (hilo_.hi >> sh));
    }
  }
};

// This is a fast implementation of 32-bit unsigned divmod borrowed from CUTLASS

/// Object to encapsulate the fast division+modulus operation.
///
/// This object precomputes two values used to accelerate the computation and is best used
/// when the divisor is a grid-invariant. In this case, it may be computed in host code and
/// marshaled along other kernel arguments using the 'Params' pattern.
///
/// Example:
///
///
///   int quotient, remainder, dividend, divisor;
///
///   FastDivmod divmod(divisor);
///
///   divmod(quotient, remainder, dividend);
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmod {
  int divisor;
  unsigned int multiplier;
  unsigned int shift_right;

  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally expensive.

  __CUDA_HD__
  FastDivmod() : divisor(0), multiplier(0), shift_right(0) {}

  __CUDA_HD__
  FastDivmod(int divisor_) : divisor(divisor_) { find_divisor(multiplier, shift_right, divisor); }

  template <typename value_t>
  __CUDA_HD__ static inline value_t clz(value_t x)
  {
    for (int i = 31; i >= 0; --i) {
      if ((1 << i) & x) {
        return 31 - i;
      }
    }
    return 32;
  }

  template <typename value_t>
  __CUDA_HD__ static inline value_t find_log2(value_t x)
  {
    int a = int(31 - clz(x));
    a += (x & (x - 1)) != 0;  // Round up, add 1 if not a power of 2.
    return a;
  }

  /**
   * Find divisor, using find_log2
   */
  __CUDA_HD__
  static inline void find_divisor(unsigned int& mul, unsigned int& shr, unsigned int denom)
  {
    if (denom == 1) {
      mul = 0;
      shr = 0;
    } else {
      unsigned int p = 31 + find_log2(denom);
      unsigned m     = unsigned(((1ull << p) + unsigned(denom) - 1) / unsigned(denom));

      mul = m;
      shr = p - 32;
    }
  }

  /**
   * Find quotient and remainder using device-side intrinsics
   */
  __CUDA_HD__
  static inline void fast_divmod(
    int& quo, int& rem, int src, int div, unsigned int mul, unsigned int shr)
  {
#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if div != 1, else simply copy the source.
    quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
    quo = (div != 1) ? static_cast<int>((std::int64_t{src} * mul) >> 32) >> shr : src;
#endif
    // The remainder.
    rem = src - (quo * div);
  }

  // For long int input
  __CUDA_HD__
  static inline void fast_divmod(
    int& quo, std::int64_t& rem, std::int64_t src, int div, unsigned int mul, unsigned int shr)
  {
#if defined(__CUDA_ARCH__)
    // Use IMUL.HI if div != 1, else simply copy the source.
    quo = (div != 1) ? __umulhi(src, mul) >> shr : src;
#else
    quo = int((div != 1) ? ((src * mul) >> 32) >> shr : src);
#endif
    // The remainder.
    rem = src - (quo * div);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  __CUDA_HD__
  inline void operator()(int& quotient, int& remainder, int dividend) const
  {
    fast_divmod(quotient, remainder, dividend, divisor, multiplier, shift_right);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  ///
  /// Simply returns the quotient
  __CUDA_HD__
  inline int divmod(int& remainder, int dividend) const
  {
    int quotient;
    fast_divmod(quotient, remainder, dividend, divisor, multiplier, shift_right);
    return quotient;
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  __CUDA_HD__
  inline void operator()(int& quotient, std::int64_t& remainder, std::int64_t dividend) const
  {
    fast_divmod(quotient, remainder, dividend, divisor, multiplier, shift_right);
  }

  /// Computes integer division and modulus using precomputed values. This is computationally
  /// inexpensive.
  __CUDA_HD__
  inline int divmod(std::int64_t& remainder, std::int64_t dividend) const
  {
    int quotient;
    fast_divmod(quotient, remainder, dividend, divisor, multiplier, shift_right);
    return quotient;
  }
};

// This is a fast implementation of 64-bit unsigned divmod borrowed from CUTLASS

/// Object to encapsulate the fast division+modulus operation for 64b integer
/// division.
///
/// This object precomputes two values used to accelerate the computation and is
/// best used when the divisor is a grid-invariant. In this case, it may be
/// computed in host code and marshaled along other kernel arguments using the
/// 'Params' pattern.
///
/// Example:
///
///
///   std::uint64_t quotient, remainder, dividend, divisor;
///
///   FastDivmodU64 divmod(divisor);
///
///   divmod(quotient, remainder, dividend);
///
///   // quotient = (dividend / divisor)
///   // remainder = (dividend % divisor)
///
struct FastDivmodU64 {
  std::uint64_t divisor;
  std::uint64_t multiplier;
  unsigned int shift_right;
  unsigned int round_up;

  //
  // Static methods
  //

  /// Computes b, where 2^b is the greatest power of two that is less than or
  /// equal to x
  __CUDA_HD__
  static inline std::uint32_t integer_log2(std::uint64_t x)
  {
    std::uint32_t n = 0;
    while (x >>= 1) {
      ++n;
    }
    return n;
  }

  /// Default ctor
  __CUDA_HD__
  FastDivmodU64() : divisor(0), multiplier(0), shift_right(0), round_up(0) {}

  /// Construct the FastDivmod object, in host code ideally.
  ///
  /// This precomputes some values based on the divisor and is computationally
  /// expensive.
  __CUDA_HD__
  FastDivmodU64(std::uint64_t divisor_)
    : divisor(divisor_), multiplier(1), shift_right(0), round_up(0)
  {
    if (divisor) {
      shift_right = integer_log2(divisor);

      if ((divisor & (divisor - 1)) == 0) {
        multiplier = 0;
      } else {
        std::uint64_t power_of_two  = (uint64_t(1) << shift_right);
        std::uint64_t multiplier_lo = uint128_t(0, power_of_two) / divisor;
        multiplier                  = uint128_t(power_of_two, power_of_two) / divisor;
        round_up                    = (multiplier_lo == multiplier ? 1 : 0);
      }
    }
  }

  /// Returns the quotient of floor(dividend / divisor)
  __CUDA_HD__
  inline std::uint64_t divide(std::uint64_t dividend) const
  {
    std::uint64_t quotient = 0;

#ifdef __CUDA_ARCH__
    std::uint64_t x = dividend;
    if (multiplier) {
      x = __umul64hi(dividend + round_up, multiplier);
    }
    quotient = (x >> shift_right);
#else
    // TODO - use proper 'fast' division here also. No reason why x86-code
    // shouldn't be optimized.
    quotient = dividend / divisor;
#endif

    return quotient;
  }

  /// Computes the remainder given a computed quotient and dividend
  __CUDA_HD__
  inline std::uint64_t modulus(std::uint64_t quotient, std::uint64_t dividend) const
  {
    return uint32_t(dividend - quotient * divisor);
  }

  /// Returns the quotient of floor(dividend / divisor) and computes the
  /// remainder
  __CUDA_HD__
  inline std::uint64_t divmod(std::uint64_t& remainder, std::uint64_t dividend) const
  {
    std::uint64_t quotient = divide(dividend);
    remainder              = modulus(quotient, dividend);
    return quotient;
  }

  /// Computes integer division and modulus using precomputed values. This is
  /// computationally inexpensive.
  __CUDA_HD__
  inline void operator()(std::uint64_t& quotient,
                         std::uint64_t& remainder,
                         std::uint64_t dividend) const
  {
    quotient = divmod(remainder, dividend);
  }
};

}  // namespace legate::detail
