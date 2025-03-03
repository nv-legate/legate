/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/tuple.h>

#include <legion.h>

#include <cstdint>
#include <string>

namespace legate::detail {

class Shape {
  enum class State : std::uint8_t {
    UNBOUND,
    BOUND,
    READY,
  };

 public:
  explicit Shape(std::uint32_t dim);
  explicit Shape(tuple<std::uint64_t>&& extents);

  [[nodiscard]] bool unbound() const;
  [[nodiscard]] bool ready() const;
  [[nodiscard]] std::uint32_t ndim() const;
  [[nodiscard]] std::size_t volume();
  [[nodiscard]] const tuple<std::uint64_t>& extents();
  [[nodiscard]] const Legion::IndexSpace& index_space();

  void set_index_space(const Legion::IndexSpace& index_space);
  void copy_extents_from(const Shape& other);

  [[nodiscard]] std::string to_string() const;

  bool operator==(Shape& other);
  bool operator!=(Shape& other);

 private:
  void ensure_binding_();

  State state_{State::UNBOUND};
  std::uint32_t dim_{};
  tuple<std::uint64_t> extents_{};
  Legion::IndexSpace index_space_{};
};

}  // namespace legate::detail

#include <legate/data/detail/shape.inl>
