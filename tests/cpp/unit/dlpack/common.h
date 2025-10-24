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
                                  legate::Half,
                                  float,
                                  double,
                                  legate::Complex<float>,
                                  legate::Complex<double>>;

using SignedIntTypes = ::testing::Types<std::int8_t, std::int16_t, std::int32_t, std::int64_t>;

inline constexpr std::array<DLDataTypeCode, 11> GetUnsupportedDataTypeCodes()
{
  return std::array<DLDataTypeCode, 11>{DLDataTypeCode::kDLFloat8_e3m4,
                                        DLDataTypeCode::kDLFloat8_e4m3,
                                        DLDataTypeCode::kDLFloat8_e4m3b11fnuz,
                                        DLDataTypeCode::kDLFloat8_e4m3fn,
                                        DLDataTypeCode::kDLFloat8_e4m3fnuz,
                                        DLDataTypeCode::kDLFloat8_e5m2,
                                        DLDataTypeCode::kDLFloat8_e5m2fnuz,
                                        DLDataTypeCode::kDLFloat8_e8m0fnu,
                                        DLDataTypeCode::kDLFloat6_e2m3fn,
                                        DLDataTypeCode::kDLFloat6_e3m2fn,
                                        DLDataTypeCode::kDLFloat4_e2m1fn};
}

inline constexpr std::array<DLDeviceType, 1> GetCPUDeviceTypes()
{
  return std::array<DLDeviceType, 1>{DLDeviceType::kDLCPU};
}

inline constexpr std::array<DLDeviceType, 3> GetGPUDeviceTypes()
{
  return std::array<DLDeviceType, 3>{
    DLDeviceType::kDLCUDA, DLDeviceType::kDLCUDAHost, DLDeviceType::kDLCUDAManaged};
}

inline constexpr std::array<DLDeviceType, 11> GetUnsupportedDeviceTypes()
{
  return std::array<DLDeviceType, 11>{DLDeviceType::kDLOpenCL,
                                      DLDeviceType::kDLVulkan,
                                      DLDeviceType::kDLMetal,
                                      DLDeviceType::kDLVPI,
                                      DLDeviceType::kDLROCM,
                                      DLDeviceType::kDLROCMHost,
                                      DLDeviceType::kDLExtDev,
                                      DLDeviceType::kDLOneAPI,
                                      DLDeviceType::kDLWebGPU,
                                      DLDeviceType::kDLHexagon,
                                      DLDeviceType::kDLMAIA};
}

inline constexpr std::array<legate::Type::Code, 3> GetUnsupportedLegateTypes()
{
  return std::array<legate::Type::Code, 3>{
    legate::Type::Code::NIL, legate::Type::Code::FIXED_ARRAY, legate::Type::Code::STRUCT};
}

template <typename T>
[[nodiscard]] DLDataTypeCode to_dlpack_code()
{
  if constexpr (std::is_same_v<T, bool>) {
    return DLDataTypeCode::kDLBool;
  } else if constexpr (std::is_integral_v<T>) {
    return std::is_signed_v<T> ? DLDataTypeCode::kDLInt : DLDataTypeCode::kDLUInt;
  } else if constexpr (std::is_floating_point_v<T>) {
    return DLDataTypeCode::kDLFloat;
  } else if constexpr (legate::type_code_of_v<T> == legate::Type::Code::FLOAT16) {
    return DLDataTypeCode::kDLBfloat;
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
        } else if constexpr (legate::type_code_of_v<T> == legate::Type::Code::FLOAT16) {
          return "float16";
        } else {
          static_assert(sizeof(T*) != sizeof(T*));  // NOLINT
        }
      }
    }
  }
};

}  // namespace dlpack_common
