/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <legate/utilities/detail/doxygen.h>

#include <cstddef>
#include <vector>

namespace legate {

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief An object representing the raw memory and strides held by a `PhysicalStore`
 */
class InlineAllocation {
 public:
  void* ptr{};                        /**< pointer to the start of the allocation */
  std::vector<std::size_t> strides{}; /**< vector of offsets into the buffer */
};

/** @} */

}  // namespace legate
