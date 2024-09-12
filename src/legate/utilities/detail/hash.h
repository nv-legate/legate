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

#include "legate/utilities/hash.h"
#include "legate/utilities/typedefs.h"

#include "legion.h"

#include <functional>
#include <tuple>
#include <type_traits>
#include <utility>

namespace legate {

template <typename T1, typename T2>
struct hasher<std::pair<T1, T2>> {
  [[nodiscard]] std::size_t operator()(const std::pair<T1, T2>& v) const noexcept
  {
    return hash_all(v.first, v.second);
  }
};

// Must be at least 1 item in the tuple
template <typename T, typename... U>
struct hasher<std::tuple<T, U...>> {
  [[nodiscard]] std::size_t operator()(const std::tuple<T, U...>& v) const noexcept
  {
    return std::apply(hash_all<T, U...>, v);
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

template <typename T>
struct hash<std::reference_wrapper<T>> {  // NOLINT(cert-dcl58-cpp)
  [[nodiscard]] std::size_t operator()(const std::reference_wrapper<T>& v) const noexcept
  {
    return std::hash<std::decay_t<T>>{}(v.get());
  }
};

}  // namespace std
