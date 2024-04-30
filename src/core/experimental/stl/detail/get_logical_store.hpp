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

#include "config.hpp"
#include "legate.h"

#include <utility>

// Include this last:
#include "prefix.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////
// get_logical_store
//   A customizable accessor for getting the underlying legate::LogicalStore from a
//   logical_store_like object.
namespace legate::stl::detail {
namespace tags {
namespace get_logical_store {

void get_logical_store();

class tag {
 public:
  [[nodiscard]] LogicalStore operator()(LogicalStore store) const noexcept { return store; }

  template <typename StoreLike>
  [[nodiscard]] auto operator()(StoreLike&& store_like) const
    noexcept(noexcept(get_logical_store(std::forward<StoreLike>(store_like))))
      -> decltype(get_logical_store(std::forward<StoreLike>(store_like)))
  {
    // Intentionally using ADL (unqualified call) here.
    return get_logical_store(std::forward<StoreLike>(store_like));
  }
};

}  // namespace get_logical_store

inline namespace obj {

inline constexpr get_logical_store::tag get_logical_store{};

}  // namespace obj
}  // namespace tags

// Fully qualify the namespace to ensure that the compiler doesn't pick some other random one
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace ::legate::stl::detail::tags::obj;

}  // namespace legate::stl::detail

#include "suffix.hpp"
