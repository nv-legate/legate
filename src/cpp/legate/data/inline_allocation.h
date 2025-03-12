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
 * @brief An object representing the raw memory and strides held by a `PhysicalStore`.
 */
class InlineAllocation {
 public:
  /**
   * @brief pointer to the start of the allocation.
   */
  void* ptr{};

  /**
   * @brief vector of byte-offsets into `ptr`.
   *
   * The offsets are given in bytes, and represent the number of bytes needed to jump to the
   * next array element in the corresponding dimension. For example:
   *
   * #. For an array whose entries are 4 bytes long, and whose shape is `(1,)`, `strides` would
   *    be `[4]` (entry `i` given by `4*i`).
   * #. For an array whose entries are 8 bytes long, and whose shape is `(1, 2,)`, `strides` would
   *    be `[16, 8]` (entry `(i, j)` given by `16*i + 8*j`).
   * #. For an array whose entries are 8 bytes long, and whose shape is `(1, 2, 3)`,
   *    strides` would be `[48, 24, 8]` (entry `(i, j, k)` given by `48*i + 24*j + 8*k`).
   */
  std::vector<std::size_t> strides{};
};

/** @} */

}  // namespace legate
