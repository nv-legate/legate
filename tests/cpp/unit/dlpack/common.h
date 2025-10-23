/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/type/complex.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/detail/dlpack/dlpack.h>
#include <legate/utilities/typedefs.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <type_traits>

namespace dlpack_common {

using AllTypes = ::testing::Types<std::int8_t,
                                  std::int16_t,
                                  std::int32_t,
                                  std::int64_t,
                                  std::uint8_t,
                                  std::uint16_t,
                                  std::uint32_t,
                                  std::uint64_t,
                                  float,
                                  double,
                                  legate::Complex<float>,
                                  legate::Complex<double>>;

using SignedIntTypes = ::testing::Types<std::int8_t, std::int16_t, std::int32_t, std::int64_t>;

template <typename T>
[[nodiscard]] DLDataTypeCode to_dlpack_code()
{
  if constexpr (std::is_same_v<T, bool>) {
    return DLDataTypeCode::kDLBool;
  } else if constexpr (std::is_integral_v<T>) {
    return std::is_signed_v<T> ? DLDataTypeCode::kDLInt : DLDataTypeCode::kDLUInt;
  } else if constexpr (std::is_floating_point_v<T>) {
    return DLDataTypeCode::kDLFloat;
  } else if constexpr (legate::is_complex_type<T>::value) {
    return DLDataTypeCode::kDLComplex;
  } else {
    static_assert(sizeof(T*) != sizeof(T*));  // NOLINT
  }
}

class NameGenerator {
 public:
  template <typename T>
  static std::string GetName(int)  // NOLINT
  {
    if constexpr (std::is_signed_v<T>) {
      if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_same_v<T, std::int8_t>) {
          return "int8";
        } else if constexpr (std::is_same_v<T, std::int16_t>) {
          return "int16";
        } else if constexpr (std::is_same_v<T, std::int32_t>) {
          return "int32";
        } else if constexpr (std::is_same_v<T, std::int64_t>) {
          return "int64";
        } else {
          static_assert(sizeof(T*) != sizeof(T*));  // NOLINT
        }
      } else {
        if constexpr (std::is_same_v<T, float>) {
          return "float";
        } else if constexpr (std::is_same_v<T, double>) {
          return "double";
        } else {
          static_assert(sizeof(T*) != sizeof(T*));  // NOLINT
        }
      }
    } else {
      if constexpr (std::is_integral_v<T>) {
        if constexpr (std::is_same_v<T, std::uint8_t>) {
          return "uint8";
        } else if constexpr (std::is_same_v<T, std::uint16_t>) {
          return "uint16";
        } else if constexpr (std::is_same_v<T, std::uint32_t>) {
          return "uint32";
        } else if constexpr (std::is_same_v<T, std::uint64_t>) {
          return "uint64";
        } else if constexpr (std::is_same_v<T, bool>) {
          return "bool";
        } else {
          static_assert(sizeof(T*) != sizeof(T*));  // NOLINT
        }
      } else {
        if constexpr (std::is_same_v<T, legate::Complex<float>>) {
          return "complex(float)";
        } else if constexpr (std::is_same_v<T, legate::Complex<double>>) {
          return "complex(double)";
        } else {
          static_assert(sizeof(T*) != sizeof(T*));  // NOLINT
        }
      }
    }
  }
};

}  // namespace dlpack_common
