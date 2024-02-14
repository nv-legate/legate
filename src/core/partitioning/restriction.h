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

#include "core/utilities/tuple.h"

namespace legate {

/**
 * @brief Enum to describe partitioning preference on dimensions of a store
 */
enum class Restriction : std::int32_t {
  ALLOW  = 0, /*!< The dimension can be partitioned */
  AVOID  = 1, /*!< The dimension can be partitioned, but other dimensions are preferred */
  FORBID = 2, /*!< The dimension must not be partitioned */
};

using Restrictions = tuple<Restriction>;

[[nodiscard]] Restriction join(Restriction lhs, Restriction rhs);

[[nodiscard]] tuple<Restriction> join(const Restrictions& lhs, const Restrictions& rhs);

void join_inplace(Restrictions& lhs, const Restrictions& rhs);

}  // namespace legate
