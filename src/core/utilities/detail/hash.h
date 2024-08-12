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

#include "core/utilities/hash.h"
#include "core/utilities/typedefs.h"

#include "legion.h"

namespace legate {

template <typename T1, typename T2>
struct hasher<std::pair<T1, T2>> {
  [[nodiscard]] std::size_t operator()(const std::pair<T1, T2>& v) const noexcept
  {
    return hash_all(v.first, v.second);
  }
};

}  // namespace legate

namespace std {

template <>
struct hash<legate::Domain> {
  [[nodiscard]] std::size_t operator()(const legate::Domain& domain) const noexcept
  {
    std::size_t result = 0;
    for (std::int32_t idx = 0; idx < 2 * domain.dim; ++idx) {
      legate::hash_combine(result, domain.rect_data[idx]);
    }
    return result;
  }
};

}  // namespace std
