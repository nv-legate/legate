/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll.h>

#include <cstdint>
#include <type_traits>

namespace common_comm {

// Type-to-CollDataType mapping
template <typename T>
struct TypeToCollDataType;

/**
 * @brief Type-to-CollDataType mapping specialization for `std::int8_t`, returns `CollInt8`.
 */
template <>
struct TypeToCollDataType<std::int8_t> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollInt8;
};

/**
 * @brief Type-to-CollDataType mapping specialization for `char`, returns `CollChar`.
 */
template <>
struct TypeToCollDataType<char> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollChar;
};

/**
 * @brief Type-to-CollDataType mapping specialization for `std::uint8_t`, returns `CollUint8`.
 */
template <>
struct TypeToCollDataType<std::uint8_t> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollUint8;
};

/**
 * @brief Type-to-CollDataType mapping specialization for `int`, returns `CollInt`.
 */
template <>
struct TypeToCollDataType<int> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollInt;
};

/**
 * @brief Type-to-CollDataType mapping specialization for `std::uint32_t`, returns `CollUint32`.
 */
template <>
struct TypeToCollDataType<std::uint32_t> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollUint32;
};

/**
 * @brief Type-to-CollDataType mapping specialization for `std::int64_t`, returns `CollInt64`.
 */
template <>
struct TypeToCollDataType<std::int64_t> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollInt64;
};

/**
 * @brief Type-to-CollDataType mapping specialization for `std::uint64_t`, returns `CollUint64`.
 */
template <>
struct TypeToCollDataType<std::uint64_t> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollUint64;
};

/**
 * @brief Type-to-CollDataType mapping specialization for `float`, returns `CollFloat`.
 */
template <>
struct TypeToCollDataType<float> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollFloat;
};

/**
 * @brief Type-to-CollDataType mapping specialization for `double`.
 */
template <>
struct TypeToCollDataType<double> {
  static constexpr auto VALUE = legate::comm::coll::CollDataType::CollDouble;
};

}  // namespace common_comm
