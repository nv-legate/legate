/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/logical_array.h>
#include <legate/utilities/detail/doxygen.h>

#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

/**
 * @file
 * @brief Interface for KVikIO I/O
 */

namespace legate {

class Shape;
class Type;

}  // namespace legate

namespace legate::experimental::io::kvikio {

/**
 * @addtogroup io-kvikio
 * @{
 */

/**
 * @brief Read a LogicalArray from a file.
 *
 * The array stored in file_path must have been written by a call to
 * `to_file(const std::filesystem::path&, const LogicalArray&)`.
 *
 * This routine expects the file to contain nothing but the raw data linearly in memory,
 * starting at offset 0. The file must contain no other metadata, padding, or other data, it
 * will be interpreted as data to be read into the store.
 *
 * @param file_path The path to the file.
 * @param type The datatype of the array.
 *
 * @return LogicalArray The loaded array.
 *
 * @throws std::system_error If `file_path` does not exist.
 *
 * @warning This API is experimental. A future release may change or remove this API without
 * warning, deprecation period, or notice. The user is nevertheless encouraged to use this API,
 * and submit any feedback to legate@nvidia.com.
 */
[[nodiscard]] LogicalArray from_file(const std::filesystem::path& file_path, const Type& type);

/**
 * @brief Write a LogicalArray to a file.
 *
 * The array must be linear, i.e. have dimension of 1.
 *
 * @param file_path The path to the file.
 * @param array The array to seralize.
 *
 * @throws std::invalid_argument If the dimension of `array` is not 1.
 *
 * @warning This API is experimental. A future release may change or remove this API without
 * warning, deprecation period, or notice. The user is nevertheless encouraged to use this API,
 * and submit any feedback to legate@nvidia.com.
 */
void to_file(const std::filesystem::path& file_path, const LogicalArray& array);

// ==========================================================================================

/**
 * @brief Load a LogicalArray from a file in tiles.
 *
 * The file must have been written by a call to `to_file()`. If `tile_start` is not given, it is
 * initialized with zeros.
 *
 * `tile_start` and `tile_shape` must have the same size.
 *
 * `array` must have the same number of dimensions as tiles. In effect `array.dim()` must equal
 * `tile_shape.size()`.
 *
 * The array shape must be divisible by the tile shape.
 *
 * Given some array stored on disk as:
 *
 * @code{.unparsed}
 * [1, 2, 3, 4, 5, 6, 7, 8, 9]
 * @endcode
 *
 * `tile_shape` sets the leaf-task launch group size. For example, `tile_shape = [3]` would
 * result in each leaf-task getting assigned a contiguous triplet of the array:
 *
 * @code{.unparsed}
 *   task_0     task_1     task_2
 * ____|____  _____|___  ____|____
 * [1, 2, 3], [4, 5, 6], [7, 8, 9]
 * @endcode
 *
 * `tile_start` is a local offset into the tile from which to begin reading. Given `tile_start
 * = [1]`, in the above example would mean that the resulting array would be read as:
 *
 * @code{.unparsed}
 * // First, split into tile_shape shapes.
 * [1, 2, 3], [4, 5, 6], [7, 8, 9]
 * // Then apply the offset (1) to each subgroup
 *    [2, 3],    [5, 6],    [8, 9]
 * @endcode
 *
 * Such that the resulting array would contain:
 *
 * @code{.unparsed}
 * [2, 3, 5, 6, 8, 9]
 * @endcode
 *
 * @param file_path The path to the dataset.
 * @param shape The shape of the resulting array.
 * @param type The datatype of the array.
 * @param tile_shape The shape of each tile.
 * @param tile_start The offsets into each tile from which to read.
 *
 * @return LogicalArray The loaded array.
 *
 * @throws std::system_error If `file_path` does not exist.
 * @throws std::invalid_argument If `tile_shape` and `tile_start` are not the same size.
 * @throws std::invalid_argument If the array dimension does not match the tile shape.
 * @throws std::invalid_argument If the array shape is not divisible by the tile shape.
 *
 * @warning This API is experimental. A future release may change or remove this API without
 * warning, deprecation period, or notice. The user is nevertheless encouraged to use this API,
 * and submit any feedback to legate@nvidia.com.
 */
[[nodiscard]] LogicalArray from_file(const std::filesystem::path& file_path,
                                     const Shape& shape,
                                     const Type& type,
                                     const std::vector<std::uint64_t>& tile_shape,
                                     std::optional<std::vector<std::uint64_t>> tile_start = {});

/**
 * @brief Write a LogicalArray to file in tiles.
 *
 * If `tile_start` is not given, it is initialized with zeros.
 *
 * `tile_start` and `tile_shape` must have the same size.
 *
 * `array` must have the same number of dimensions as tiles. In effect `array.dim()` must equal
 * `tile_shape.size()`.
 *
 * The array shape must be divisible by the tile shape.
 *
 * See `from_file()` for further discussion on the arguments.
 *
 * @param file_path The base path of the dataset to write.
 * @param array The array to serialize.
 * @param tile_shape The shape of the tiles.
 * @param tile_start The offsets into each tile from which to write.
 *
 * @throws std::invalid_argument If `tile_shape` and `tile_start` are not the same size.
 * @throws std::invalid_argument If the array dimension does not match the tile shape.
 * @throws std::invalid_argument If the array shape is not divisible by the tile shape.
 *
 * @warning This API is experimental. A future release may change or remove this API without
 * warning, deprecation period, or notice. The user is nevertheless encouraged to use this API,
 * and submit any feedback to legate@nvidia.com.
 */
void to_file(const std::filesystem::path& file_path,
             const LogicalArray& array,
             const std::vector<std::uint64_t>& tile_shape,
             std::optional<std::vector<std::uint64_t>> tile_start = {});

// ==========================================================================================

/**
 * @brief Load a LogicalArray from a file in tiles.
 *
 * `array` must have the same number of dimensions as tiles. In effect `array.dim()` must equal
 * `tile_shape.size()`.
 *
 * This routine should be used if each leaf task in a tile should read from a potentially
 * non-uniform offset than the others. If the offset is uniform (i.e. can be deduced by the
 * leaf task index, and the tile shape), then `from_file()` should be preferred.
 *
 * For example, given some array (of int32's) stored on disk as:
 *
 * @code{.unparsed}
 * [1, 2, 3, 4, 5, 6, 7, 8, 9]
 * @endcode
 *
 * `tile_shape` sets the leaf-task launch group size. For example, `tile_shape = {3}` would
 * result in each leaf-task getting assigned a contiguous triplet of the array:
 *
 * @code{.unparsed}
 *   task_0     task_1     task_2
 * ____|____  _____|___  ____|____
 * [1, 2, 3], [4, 5, 6], [7, 8, 9]
 * @endcode
 *
 * It also sets the number of elements to read. Each leaf-task will read
 * `tile_shape.volume() * type.size()` bytes from the file.
 *
 * `offsets` encodes the per-leaf-task global offset in bytes into the array for each
 * tile. Crucially, these offsets need not (and by definition shall not) be the same for each
 * leaf task. For example, assuming `sizeof(std::int32_t) = 4`:
 *
 * @code{.cpp}
 * std::vector<std::uint64_t> offsets = {
 *   // task_0 reads from byte index 0 of the file (i.e. starting from element 0)
 *   0,
 *   // task_1 reads from byte index 4 * 3 = 12 of the file (i.e. starting from element 4)
 *   3 * sizeof(std::int32_t),
 *   // task_2 reads from byte index 4 * 7 = 28 of the file (i.e. starting from element 8)
 *   7 * sizeof(std::int32_t),
 * };
 * @endcode
 *
 * Note how the final offset is arbitrary. If the offsets were uniform, it would start from
 * element 7. The resulting array would then contain:
 *
 * @code{.unparsed}
 * [1, 2, 3, 4, 5, 6, 8, 9]
 * @endcode
 *
 * If the data is multi-dimensional, the task IDs for the purposes of indexing into `offsets`
 * are linearized in C order. For example, if we have 2x2 tiles (`tile_shape = {2, 2}`), the
 * task IDs would be linearized as follows:
 *
 * @code{.unparsed}
 * (0, 0) -> 0
 * (0, 1) -> 1
 * (1, 0) -> 2
 * (1, 1) -> 3
 * @endcode
 *
 * @param file_path The path to the file to read.
 * @param shape The shape of the resulting array.
 * @param type The datatype of the array.
 * @param offsets The per-leaf-task global offsets (in bytes) into the file from which to read.
 * @param tile_shape The shape of each tile.
 *
 * @return LogicalArray The loaded array.
 *
 * @throws std::system_error If `file_path` does not exist.
 * @throws std::invalid_argument If `offsets.size()` does not equal the number of partitioned array
 * tiles.
 *
 * @warning This API is experimental. A future release may change or remove this API without
 * warning, deprecation period, or notice. The user is nevertheless encouraged to use this API,
 * and submit any feedback to legate@nvidia.com.
 */
[[nodiscard]] LogicalArray from_file_by_offsets(const std::filesystem::path& file_path,
                                                const Shape& shape,
                                                const Type& type,
                                                const std::vector<std::uint64_t>& offsets,
                                                const std::vector<std::uint64_t>& tile_shape);

/** @} */

}  // namespace legate::experimental::io::kvikio
