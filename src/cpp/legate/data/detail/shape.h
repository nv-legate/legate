/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <string>

namespace legate::detail {

class Shape {
  enum class State : std::uint8_t {
    UNBOUND,
    BOUND,
  };

 public:
  explicit Shape(std::uint32_t dim);
  explicit Shape(SmallVector<std::uint64_t, LEGATE_MAX_DIM>&& extents);

  [[nodiscard]] bool unbound() const;
  [[nodiscard]] std::uint32_t ndim() const;
  [[nodiscard]] std::size_t volume();
  [[nodiscard]] Span<const std::uint64_t> extents();

  void copy_extents_from(const Shape& other);

  /**
   * @brief Binds this (previously unbound) shape to a concrete set of extents.
   *
   * Used when a task binds a buffer to an unbound store: the store's shape, which up to this
   * point only knew its dimensionality, becomes fully known (BOUND) with the given extents. It
   * is legal to call this on a shape in any state as long as the dimensionality matches; the
   * shape transitions directly to BOUND.
   *
   * @param extents The now-known extents of the shape.
   */
  void bind_extents(SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents);

  [[nodiscard]] std::string to_string() const;

  bool operator==(Shape& other);
  bool operator!=(Shape& other);

 private:
  void ensure_binding_();

  State state_{State::UNBOUND};
  std::uint32_t dim_{};
  SmallVector<std::uint64_t, LEGATE_MAX_DIM> extents_{};
};

}  // namespace legate::detail

#include <legate/data/detail/shape.inl>
