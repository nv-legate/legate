/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
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
