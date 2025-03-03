/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/typedefs.h>

#include <string>

namespace legate {

/**
 * @addtogroup util
 * @{
 */

/**
 * @file
 * @brief Debugging utilities
 */

/**
 * @brief Converts the dense array into a string
 *
 * @param base Array to convert
 * @param extents Extents of the array
 * @param strides Strides for dimensions
 *
 * @return A string expressing the contents of the array
 */
template <typename T, int DIM>
[[nodiscard]] std::string print_dense_array(const T* base,
                                            const Point<DIM>& extents,
                                            const std::size_t (&strides)[DIM]);
/**
 * @brief Converts the dense array into a string using an accessor
 *
 * @param accessor Accessor to an array
 * @param rect Sub-rectangle within which the elements should be retrieved
 *
 * @return A string expressing the contents of the array
 */
template <int DIM, typename ACC>
[[nodiscard]] std::string print_dense_array(const ACC& accessor, const Rect<DIM>& rect);
// TODO(mpapadakis): Disabled while we find a workaround for operator<< missing for
// cuda::std::complex, see legate.internal#475
// /**
//  * @ingroup util
//  * @brief Converts the store to a string
//  *
//  * @param store Store to convert
//  *
//  * @return A string expressing the contents of the store
//  */
// [[nodiscard]] std::string print_dense_array(const PhysicalStore& store);

/** @} */

}  // namespace legate

#include <legate/utilities/debug.inl>
