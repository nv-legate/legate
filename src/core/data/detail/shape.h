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

#include "legion.h"

#include <cstdint>
#include <string>

namespace legate::detail {

class Shape {
 private:
  enum class State : uint64_t {
    UNBOUND,
    BOUND,
    READY,
  };

 public:
  explicit Shape(uint32_t dim);
  explicit Shape(tuple<uint64_t>&& extents);

  [[nodiscard]] bool ready() const;
  [[nodiscard]] uint32_t ndim() const;
  [[nodiscard]] size_t volume();
  [[nodiscard]] const tuple<uint64_t>& extents();
  [[nodiscard]] const Legion::IndexSpace& index_space();

  void set_index_space(const Legion::IndexSpace& index_space);

  [[nodiscard]] std::string to_string() const;

  bool operator==(Shape& other);
  bool operator!=(Shape& other);

 private:
  void ensure_binding();

  State state_{State::UNBOUND};
  uint32_t dim_{};
  tuple<uint64_t> extents_{};
  Legion::IndexSpace index_space_{};
};

}  // namespace legate::detail

#include "core/data/detail/shape.inl"
