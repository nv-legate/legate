/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/data/physical_store.h"
#include "core/utilities/typedefs.h"

#include <string>

/** @defgroup util Utilities
 */

/**
 * @file
 * @brief Debugging utilities
 */
namespace legate {

/**
 * @ingroup util
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
                                            std::size_t strides[DIM]);
/**
 * @ingroup util
 * @brief Converts the dense array into a string using an accessor
 *
 * @param accessor Accessor to an array
 * @param rect Sub-rectangle within which the elements should be retrieved
 *
 * @return A string expressing the contents of the array
 */
template <int DIM, typename ACC>
[[nodiscard]] std::string print_dense_array(ACC accessor, const Rect<DIM>& rect);
/**
 * @ingroup util
 * @brief Converts the store to a string
 *
 * @param store Store to convert
 *
 * @return A string expressing the contents of the store
 */
[[nodiscard]] std::string print_dense_array(const PhysicalStore& store);

}  // namespace legate

#include "core/utilities/debug.inl"
