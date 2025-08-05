/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/exception.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/shared_ptr.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {

/**
 * @addtogroup task
 * @{
 */

/**
 * @brief An exception class used to model exceptions thrown from Python.
 *
 * Any client that needs to catch a C++ exception during task execution and have it rethrown
 * on the launcher side should wrap that C++ exception with a `TaskException`. If a Python task
 * throws a Python exception, it is automatically converted to, and stored within, a
 * PythonTaskException.
 *
 * @see TaskException
 */
class LEGATE_EXPORT PythonTaskException : public TaskException {
  using base_type = TaskException;

 public:
  PythonTaskException() = delete;

  /**
   * @brief Construct a PythonTaskException from a pickled Python exception.
   *
   * @param size The size, in bytes, of the given buffer.
   * @param buf The pointer to the pickled Python exception object.
   */
  PythonTaskException(std::uint64_t size, SharedPtr<const std::byte[]> buf);

  /**
   * @brief Return a pointer to the Python exception object.
   *
   * @return A pointer to the exception object.
   */
  [[nodiscard]] const std::byte* data() const noexcept;

  /**
   * @brief Return the size (in bytes) of the held buffer.
   *
   * @return The size of the held buffer.
   */
  [[nodiscard]] std::uint64_t size() const noexcept;

 private:
  std::uint64_t size_{};
  SharedPtr<const std::byte[]> bytes_{};
};

/** @} */

}  // namespace legate::detail

#include <legate/task/detail/exception.inl>
