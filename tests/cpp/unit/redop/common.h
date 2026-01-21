/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/redop/redop.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <utilities/utilities.h>

namespace redop_test {

// Common test fixture
using RedopUnit = DefaultFixture;

// Bit pattern constants for bitwise reduction tests
constexpr int K_BITS_1010 = 0b1010;
constexpr int K_BITS_0101 = 0b0101;
constexpr int K_BITS_1111 = 0b1111;
constexpr int K_BITS_1100 = 0b1100;
constexpr int K_BITS_0110 = 0b0110;
constexpr int K_BITS_00FF = 0xFF;

// Numeric constants for reduction tests
constexpr int K_INT_10        = 10;
constexpr float K_FLOAT_2_0   = 2.0F;
constexpr float K_FLOAT_3_0   = 3.0F;
constexpr float K_FLOAT_4_0   = 4.0F;
constexpr float K_FLOAT_5_0   = 5.0F;
constexpr float K_FLOAT_5_5   = 5.5F;
constexpr float K_FLOAT_10_0  = 10.0F;
constexpr double K_DOUBLE_3_0 = 3.0;
constexpr double K_DOUBLE_4_0 = 4.0;
constexpr double K_DOUBLE_5_5 = 5.5;

// Integer and floating-point types for Sum/Prod/Max/Min
using NumericTypes = ::testing::Types<std::int8_t,
                                      std::int16_t,
                                      std::int32_t,
                                      std::int64_t,
                                      std::uint8_t,
                                      std::uint16_t,
                                      std::uint32_t,
                                      std::uint64_t,
                                      float,
                                      double>;

// Integer types for bitwise operations (Or/And/XOR)
using IntegerTypes = ::testing::Types<std::int8_t,
                                      std::int16_t,
                                      std::int32_t,
                                      std::int64_t,
                                      std::uint8_t,
                                      std::uint16_t,
                                      std::uint32_t,
                                      std::uint64_t>;

}  // namespace redop_test
