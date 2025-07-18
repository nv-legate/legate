/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/type/types.h>
#include <legate/utilities/detail/dlpack/dlpack.h>

#include <fmt/base.h>

#include <cstddef>
#include <cstdint>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}  // namespace legate::mapping

namespace legate::detail {

/**
 * @brief Convert the DLPack device type to Legate StoreTarget.
 *
 * @param dtype The device type to convert.
 *
 * @return The store target.
 *
 * @throw std::invalid_argument If the conversion is not supported.
 */
[[nodiscard]] mapping::StoreTarget to_store_target(DLDeviceType dtype);

/**
 * @brief Convert a Legate type code to DLPack format.
 *
 * @param code The legate type to convert.
 *
 * @return The dlpack code.
 *
 * @throw std::invalid_argument If the conversion is not supported.
 */
[[nodiscard]] DLDataTypeCode to_dlpack_type(legate::Type::Code code);

/**
 * @brief Compute the size (in bytes) of the data array of a DLTensor.
 *
 * @param tensor The tensor to compute the size of.
 *
 * @return The size.
 */
[[nodiscard]] std::size_t dl_tensor_size(const DLTensor& tensor);

}  // namespace legate::detail

namespace fmt {

template <>
struct formatter<DLDeviceType> : formatter<string_view> {
  format_context::iterator format(DLDeviceType dtype, format_context& ctx) const;
};

template <>
struct formatter<DLDataTypeCode> : formatter<string_view> {
  format_context::iterator format(DLDataTypeCode code, format_context& ctx) const;
};

}  // namespace fmt
