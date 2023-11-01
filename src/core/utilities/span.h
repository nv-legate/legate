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

#include <cstddef>

/**
 * @file
 * @brief Class definition for legate::Span
 */

namespace legate {

/**
 * @ingroup data
 * @brief A simple span implementation used in Legate. Should eventually be replaced with
 * std::span once we bump up the C++ standard version to C++20
 */
template <typename T>
class Span {
 public:
  Span() = default;
  /**
   * @brief Creates a span with an existing pointer and a size.
   *
   * The caller must guarantee that the allocation is big enough (i.e., bigger than or
   * equal to `sizeof(T) * size`) and that the allocation is alive while the span is alive.
   *
   * @param data Pointer to the data
   * @param size Number of elements
   */
  Span(T* data, std::size_t size);
  /**
   * @brief Returns the number of elements
   *
   * @return The number of elements
   */
  [[nodiscard]] std::size_t size() const;

  [[nodiscard]] decltype(auto) operator[](std::size_t pos) const;
  /**
   * @brief Returns the pointer to the first element
   *
   * @return Pointer to the first element
   */
  [[nodiscard]] const T* begin() const;
  /**
   * @brief Returns the pointer to the end of allocation
   *
   * @return Pointer to the end of allocation
   */
  [[nodiscard]] const T* end() const;
  /**
   * @brief Slices off the first `off` elements. Passing an `off` greater than
   * the size will fail with an assertion failure.
   *
   * @param off Number of elements to skip
   *
   * @return A span for range `[off, size())`
   */
  [[nodiscard]] Span subspan(std::size_t off);
  /**
   * @brief Returns a `const` pointer to the data
   *
   * @return Pointer to the data
   */
  [[nodiscard]] const T* ptr() const;

 private:
  T* data_{};
  std::size_t size_{};
};

}  // namespace legate

#include "core/utilities/span.inl"
