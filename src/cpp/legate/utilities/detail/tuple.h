/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {

// Commonly used conversion routines for tuples

/**
 * @brief Converts a shape to a domain.
 *
 * @note If the shape is empty, a 1D {0, 0} domain is returned.
 *
 * @param shape The shape to convert.
 *
 * @return The domain.
 */
[[nodiscard]] Rect to_domain(Span<const std::uint64_t> shape);

/**
 * @brief Converts a shape expressed as a `Point` to a domain `[0, shape - 1]`.
 *
 * @note If the shape is empty (0-dimensional), a 1D {0, 0} domain is returned.
 *
 * @param shape The shape to convert.
 *
 * @return The domain.
 */
[[nodiscard]] Rect to_domain(const Point& shape);

/**
 * @brief Converts a shape to a domain.
 *
 * @note If the shape is empty, an empty domain {} is returned.
 *
 * @param shape The shape to convert.
 *
 * @return The domain.
 */
[[nodiscard]] Rect to_maybe_empty_domain(Span<const std::uint64_t> shape);

[[nodiscard]] LEGATE_PYTHON_EXPORT Point to_domain_point(Span<const std::uint64_t> shape);

[[nodiscard]] Point to_domain_point(Span<const std::int64_t> shape);

[[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> from_domain(const Rect& domain);

// These are forward declared in tuple.h to avoid including this header
// NOLINTBEGIN(readability-redundant-declaration)
[[noreturn]] LEGATE_EXPORT void throw_invalid_tuple_sizes(std::size_t lhs_size,
                                                          std::size_t rhs_size);
// NOLINTEND(readability-redundant-declaration)

}  // namespace legate::detail
