/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/experimental/stl/detail/config.hpp>

#include <utility>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

////////////////////////////////////////////////////////////////////////////////////////////////////
// get_logical_store
//   A customizable accessor for getting the underlying legate::LogicalStore from a
//   logical_store_like object.
namespace legate::experimental::stl::detail {
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

inline constexpr get_logical_store::tag
  get_logical_store{};  // NOLINT(readability-identifier-naming)

}  // namespace obj
}  // namespace tags

// Fully qualify the namespace to ensure that the compiler doesn't pick some other random one
// NOLINTNEXTLINE(google-build-using-namespace)
using namespace ::legate::experimental::stl::detail::tags::obj;

}  // namespace legate::experimental::stl::detail

#if LEGATE_DEFINED(LEGATE_DOXYGEN)
namespace legate::experimental::stl {
/**
 * @brief Get the logical store from a logical store-like object.
 *
 * @see @c logical_store_like
 * @ingroup stl-concepts
 */
template <typename StoreLike>
LogicalStore get_logical_store(StoreLike&& store_like);
}  // namespace legate::experimental::stl
#endif

#include <legate/experimental/stl/detail/suffix.hpp>
