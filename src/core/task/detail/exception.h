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

#include "core/task/exception.h"
#include "core/utilities/shared_ptr.h"

#include <cstdint>

namespace legate::detail {

/**
 * @ingroup task
 * @brief An exception class used to model exceptions thrown from Python.
 *
 * Any client that needs to catch a C++ exception during task execution and have it rethrown
 * on the launcher side should wrap that C++ exception with a `TaskException`. If a Python task
 * throws a Python exception, it is automatically converted to, and stored within, a
 * PythonTaskException.
 *
 * @see TaskException
 */
class PythonTaskException : public TaskException {
  using base_type = TaskException;

 public:
  PythonTaskException() = delete;

  /**
   * @brief Construct a PythonTaskException from a pickled Python exception.
   *
   * @param size The size, in bytes, of the given buffer.
   * @param buf The pointer to the pickled Python exception object.
   */
  PythonTaskException(std::uint64_t size, SharedPtr<const char[]> buf);

  /**
   * @brief Return a pointer to the Python exception object.
   *
   * @return A pointer to the exception object.
   */
  [[nodiscard]] const char* data() const noexcept;

  /**
   * @brief Return the size (in bytes) of the held buffer.
   *
   * @return The size of the held buffer.
   */
  [[nodiscard]] std::uint64_t size() const noexcept;

 private:
  std::uint64_t size_{};
  SharedPtr<const char[]> bytes_{};
};

}  // namespace legate::detail

#include "core/task/detail/exception.inl"
