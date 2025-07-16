/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

[[nodiscard]] Domain to_domain(Span<const std::uint64_t> shape);

[[nodiscard]] DomainPoint to_domain_point(Span<const std::uint64_t> shape);

[[nodiscard]] SmallVector<std::uint64_t, LEGATE_MAX_DIM> from_domain(const Domain& domain);

// These are forward declared in tuple.h to avoid including this header
// NOLINTBEGIN(readability-redundant-declaration)
[[noreturn]] void throw_invalid_tuple_sizes(std::size_t lhs_size, std::size_t rhs_size);
// NOLINTEND(readability-redundant-declaration)

}  // namespace legate::detail
