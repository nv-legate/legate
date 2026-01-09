/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/logical_array.h>
#include <legate/utilities/detail/doxygen.h>

#include <filesystem>
#include <string_view>

/**
 * @file
 * @brief Interface for HDF5 I/O
 */

namespace legate::io::hdf5 {

/**
 * @addtogroup io-hdf5
 * @{
 */

/**
 * @brief An exception thrown when a HDF5 datatype could not be converted to a Type.
 */
class LEGATE_EXPORT UnsupportedHDF5DataTypeError : public std::invalid_argument {
 public:
  using std::invalid_argument::invalid_argument;
};

/**
 * @brief An exception thrown when an invalid dataset is encountered in an HDF5 file.
 */
class LEGATE_EXPORT InvalidDataSetError : public std::invalid_argument {
 public:
  /**
   * @brief Construct an InvalidDataSetError
   *
   * @param what The exception string to forward to the constructor of std::invalid_argument.
   * @param path The path to the HDF5 file containing the dataset.
   * @param dataset_name The name of the offending dataset.
   */
  InvalidDataSetError(const std::string& what,
                      std::filesystem::path path,
                      std::string dataset_name);

  /**
   * @brief Get the path to the file containing the dataset.
   *
   * @return The path to the file containing the dataset.
   */
  [[nodiscard]] const std::filesystem::path& path() const noexcept;

  /**
   * @brief Get the name of the dataset.
   *
   * @return The name of the dataset.
   */
  [[nodiscard]] std::string_view dataset_name() const noexcept;

 private:
  std::filesystem::path path_{};
  std::string dataset_name_{};
};

/**
 * @brief Load a HDF5 dataset into a LogicalArray.
 *
 * @param file_path The path to the file to load.
 * @param dataset_name The name of the HDF5 dataset to load from the file.
 *
 * @return LogicalArray The loaded array.
 *
 * @throws std::system_error If file_path does not exist.
 * @throws UnusupportedHDF5DataType If the data type cannot be converted to a Type.
 * @throws InvalidDataSetError If the dataset is invalid, or is not found.
 */
[[nodiscard]] LEGATE_EXPORT LogicalArray from_file(const std::filesystem::path& file_path,
                                                   std::string_view dataset_name);

/**
 * @brief Write a LogicalArray to disk as a HDF5 dataset.
 *
 * If `file_path` already exists at the time of writing, the file will be overwritten.
 *
 * `file_path` may be absolute or relative. If it is relative, it will be written relative to
 * the current working directory at the time of this function call.
 *
 * `file_path` may not fully exist at the time of this function call. Any missing directories
 * are created (with the same permissions and properties of the current process) before tasks
 * are launched. However, no protection is provided if those directories are later deleted
 * before the task executes -- the tasks assume these directories exist when they execute.
 *
 * `array` must not be unbound.
 *
 * @param array The array to store.
 * @param file_path The resulting HDF5 file.
 * @param dataset_name The HDF5 dataset name to store the array under. See
 * https://support.hdfgroup.org/documentation/hdf5/latest/_h5_d__u_g.html for further
 * discussion on datasets.
 *
 * @throw std::invalid_argument If `file_path` would not be a valid path name, for example if
 * it is a directory name. Generally speaking this means that it must be of the form
 * `/path/to/file.h5`.
 */
LEGATE_EXPORT void to_file(const LogicalArray& array,
                           std::filesystem::path file_path,
                           std::string_view dataset_name);

/** @} */

}  // namespace legate::io::hdf5
