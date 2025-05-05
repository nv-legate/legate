/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>
#include <legate/utilities/tuple.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace legate::detail {

// Commonly used conversion routines for tuples

[[nodiscard]] Domain to_domain(Span<const std::uint64_t> shape);

[[nodiscard]] Domain to_domain(const tuple<std::uint64_t>& shape);

[[nodiscard]] DomainPoint to_domain_point(const tuple<std::uint64_t>& shape);

[[nodiscard]] tuple<std::uint64_t> from_domain(const Domain& domain);

// These are forward declared in tuple.h to avoid including this header
// NOLINTBEGIN(readability-redundant-declaration)
void assert_valid_mapping(std::size_t tuple_size, const std::vector<std::int32_t>& mapping);
[[noreturn]] void throw_invalid_tuple_sizes(std::size_t lhs_size, std::size_t rhs_size);
void assert_in_range(std::size_t tuple_size, std::int32_t pos);
// NOLINTEND(readability-redundant-declaration)

}  // namespace legate::detail
