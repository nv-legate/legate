/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/type/half.h>

#include <cstdint>
#include <cstring>  // memcpy

namespace legate {

#if LEGATE_DEFINED(LEGATE_DEFINED_HALF)
// These functions were copied from Legion (and cleaned up), but they appear to be based on
// this stackoverflow answer: https://stackoverflow.com/a/76816560/13615317

namespace {

// NOLINTBEGIN(readability-magic-numbers)
[[nodiscard]] std::uint16_t convert_float_to_halfint(float a) noexcept
{
  std::uint32_t ia = 0;
  static_assert(sizeof(ia) == sizeof(a), "half size mismatch");
  std::memcpy(&ia, &a, sizeof(a));
  std::uint16_t ir = (ia >> 16) & 0x8000;

  if ((ia & 0x7f800000) == 0x7f800000) {
    if ((ia & 0x7fffffff) == 0x7f800000) {
      ir |= 0x7c00;  // infinity
    } else {
      ir = 0x7fff;  // canonical NaN
    }
  } else if ((ia & 0x7f800000) >= 0x33000000) {
    const auto shift = static_cast<std::int32_t>((ia >> 23) & 0xff) - 127;

    if (shift > 15) {
      ir |= 0x7c00;  // infinity
    } else {
      ia = (ia & 0x007fffff) | 0x00800000;  // extract mantissa
      if (shift < -14) {                    // denormal
        ir |= ia >> (-1 - shift);
        ia = ia << (32 - (-1 - shift));
      } else {  // normal
        ir |= ia >> (24 - 11);
        ia = ia << (32 - (24 - 11));
        ir = static_cast<std::uint16_t>(ir + ((14 + shift) << 10));
      }
      // IEEE-754 round to nearest of even
      if ((ia > 0x80000000) || ((ia == 0x80000000) && (ir & 1))) {
        ++ir;
      }
    }
  }

  return ir;
}

[[nodiscard]] float convert_halfint_to_float(std::uint16_t x) noexcept
{
  const std::int32_t sign = ((x >> 15) & 1);
  std::int32_t exp        = ((x >> 10) & 0x1f);
  std::int32_t mantissa   = (x & 0x3ff);
  std::uint32_t f         = 0;

  if (exp > 0 && exp < 31) {
    // normal
    exp += 112;
    f = static_cast<std::uint32_t>((sign << 31) | (exp << 23) | (mantissa << 13));
  } else if (exp == 0) {
    if (mantissa) {
      // subnormal
      exp += 113;
      while ((mantissa & (1 << 10)) == 0) {
        mantissa <<= 1;
        --exp;
      }
      mantissa &= 0x3ff;
      f = static_cast<std::uint32_t>((sign << 31) | (exp << 23) | (mantissa << 13));
    } else {
      // zero
      f = 0;
    }
  } else if (exp == 31) {
    if (mantissa) {
      f = 0x7fffffff;  // not a number
    } else {
      f = static_cast<std::uint32_t>((0xff << 23) | (sign << 31));  //  inf
    }
  }

  float result = 0.F;
  static_assert(sizeof(result) == sizeof(f), "half size mismatch");
  std::memcpy(&result, &f, sizeof(f));
  return result;
}

// NOLINTEND(readability-magic-numbers)

}  // namespace

Half::Half(float a) noexcept : Half{convert_float_to_halfint(a)} {}

Half::Half(double a) noexcept : Half{static_cast<float>(a)} {}

Half::Half(int a) noexcept : Half{static_cast<float>(a)} {}

Half::Half(coord_t a) noexcept : Half{static_cast<float>(a)} {}

Half::Half(std::size_t a) noexcept : Half{static_cast<float>(a)} {}

Half& Half::operator=(float rhs) noexcept
{
  repr_ = convert_float_to_halfint(rhs);
  return *this;
}

Half& Half::operator=(double rhs) noexcept
{
  *this = static_cast<float>(rhs);
  return *this;
}

Half& Half::operator=(int rhs) noexcept
{
  *this = static_cast<float>(rhs);
  return *this;
}

Half& Half::operator=(coord_t rhs) noexcept
{
  *this = static_cast<float>(rhs);
  return *this;
}

Half& Half::operator=(std::size_t rhs) noexcept
{
  *this = static_cast<float>(rhs);
  return *this;
}

Half::operator float() const noexcept { return convert_halfint_to_float(repr_); }

Half::operator double() const noexcept { return static_cast<float>(*this); }

Half& Half::operator+=(const Half& rhs) noexcept
{
  *this = static_cast<float>(*this) + static_cast<float>(rhs);
  return *this;
}

Half& Half::operator-=(const Half& rhs) noexcept
{
  *this = static_cast<float>(*this) - static_cast<float>(rhs);
  return *this;
}

Half& Half::operator*=(const Half& rhs) noexcept
{
  *this = static_cast<float>(*this) * static_cast<float>(rhs);
  return *this;
}

Half& Half::operator/=(const Half& rhs) noexcept
{
  *this = static_cast<float>(*this) / static_cast<float>(rhs);
  return *this;
}

// ==========================================================================================

Half operator-(const Half& a) noexcept { return Half{-static_cast<float>(a)}; }

// ------------------------------------------------------------------------------------------

Half operator+(const Half& a, const Half& b) noexcept
{
  return Half{static_cast<float>(a) + static_cast<float>(b)};
}

Half operator-(const Half& a, const Half& b) noexcept
{
  return Half{static_cast<float>(a) - static_cast<float>(b)};
}

Half operator*(const Half& a, const Half& b) noexcept
{
  return Half{static_cast<float>(a) * static_cast<float>(b)};
}

Half operator/(const Half& a, const Half& b) noexcept
{
  return Half{static_cast<float>(a) / static_cast<float>(b)};
}

// ------------------------------------------------------------------------------------------

bool operator<(const Half& a, const Half& b) noexcept
{
  return static_cast<float>(a) < static_cast<float>(b);
}

bool operator<=(const Half& a, const Half& b) noexcept { return !(b < a); }

bool operator>(const Half& a, const Half& b) noexcept { return b < a; }

bool operator>=(const Half& a, const Half& b) noexcept { return !(a < b); }
#endif

}  // namespace legate
