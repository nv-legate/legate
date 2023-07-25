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

#include <stdint.h>
#include <exception>
#include <string>

/**
 * @file
 * @brief Class definition for legate::TaskException
 */

namespace legate {

/**
 * @ingroup task
 * @brief An exception class used in cross language exception handling
 *
 * Any client that needs to catch a C++ exception during task execution and have it rethrown
 * on the launcher side should wrap that C++ exception with a `TaskException`. In case the
 * task can raise more than one type of exception, they are distinguished by integer ids;
 * the launcher is responsible for enumerating a list of all exceptions that can be raised
 * and the integer ids are positions in that list.
 */
class TaskException : public std::exception {
 public:
  /**
   * @brief Constructs a `TaskException` object with an exception id and an error message.
   * The id must be a valid index for the list of exceptions declared by the launcher.
   *
   * @param index Exception id
   * @param error_message Error message
   */
  TaskException(int32_t index, const std::string& error_message)
    : index_(index), error_message_(error_message)
  {
  }

  /**
   * @brief Constructs a `TaskException` object with an error message. The exception id
   * is set to 0.
   *
   * @param error_message Error message
   */
  TaskException(const std::string& error_message) : index_(0), error_message_(error_message) {}

 public:
  virtual const char* what() const throw() { return error_message_.c_str(); }

 public:
  /**
   * @brief Returns the exception id
   *
   * @return The exception id
   */
  int32_t index() const { return index_; }
  /**
   * @brief Returns the error message
   *
   * @return The error message
   */
  const std::string& error_message() const { return error_message_; }

 private:
  int32_t index_{-1};
  std::string error_message_;
};

}  // namespace legate
