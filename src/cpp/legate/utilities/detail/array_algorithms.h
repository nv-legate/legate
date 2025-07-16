/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/span.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {

/**
 * @brief Compute the volume of an array.
 *
 * @param container The array.
 *
 * @return The volume of the array.
 */
template <typename T>
[[nodiscard]] std::size_t array_volume(const T& container);

// NOLINTBEGIN(readability-redundant-declaration)
/**
 * @brief Detect if the specified mapping is injective for a container of a particular size..
 *
 * @param container_size The size of the container to be mapped.
 * @param mapping The mapping to apply.
 *
 * @throws std::out_of_range If the mapping and container size mismatch, or the mapping
 * contains indices outside of [0, container_size).
 * @throws std::invalid_argument If the mapping contains duplicate elements.
 */
void assert_valid_mapping(std::size_t container_size, Span<const std::int32_t> mapping);
// NOLINTEND(readability-redundant-declaration)

/**
 * @brief Create a copy of the container my mapping the current values via the supplied mapping.
 *
 * @param container The array to copy.
 * @param mapping The index mapping.
 *
 * @return The mapped array.
 */
template <typename T>
[[nodiscard]] T array_map(const T& container, Span<const std::int32_t> mapping);

/**
 * @brief Create a copy of a specific container type by mapping the current values via the
 * supplied mapping.
 *
 * @tparam U the type of the new container.
 *
 * @param container The span to copy from.
 * @param mapping The index mapping.
 *
 * @return The mapped array.
 */
template <typename U, typename T>
[[nodiscard]] U array_map(Span<const T> container, Span<const std::int32_t> mapping);

/**
 * @brief Determine whether a predicate `func` is true for all pair-wise elements in the arrays.
 *
 * @param func The predicate function.
 * @param arr The first array.
 * @param rest The remaining arrays.
 *
 * @return `true` if `func` is true for all `func(arr[i], rest[i]...)` entries, `false`
 * otherwise.
 */
template <typename F, typename T, typename... Tn>
[[nodiscard]] bool array_all_of(F&& func, const T& arr, const Tn&... rest);

/**
 * @brief Assert that a position is within th bounds of an array.
 *
 * @param container_size The size of the array.
 * @param pos The position to check.
 *
 * @throws std::out_of_range If the position is not in the array.
 */
void assert_in_range(std::size_t container_size,  // NOLINT(readability-redundant-declaration)
                     std::int64_t pos);

}  // namespace legate::detail

#include <legate/utilities/detail/array_algorithms.inl>
