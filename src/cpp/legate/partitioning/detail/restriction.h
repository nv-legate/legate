/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/small_vector.h>

#include <cstdint>

namespace legate::detail {

/**
 * @brief Enum to describe partitioning preference on dimensions of a store
 */
enum class Restriction : std::uint8_t {
  ALLOW  = 0, /*!< The dimension can be partitioned */
  AVOID  = 1, /*!< The dimension can be partitioned, but other dimensions are preferred */
  FORBID = 2, /*!< The dimension must not be partitioned */
};

using Restrictions = SmallVector<Restriction>;

[[nodiscard]] Restriction join(Restriction lhs, Restriction rhs);

[[nodiscard]] Restrictions join(const Restrictions& lhs, const Restrictions& rhs);

void join_inplace(Restrictions& lhs, const Restrictions& rhs);

}  // namespace legate::detail
