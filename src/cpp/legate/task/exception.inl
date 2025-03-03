/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/exception.h>

#include <utility>

namespace legate {

inline TaskException::TaskException(std::int32_t index, std::string error_message)
  : index_{index}, error_message_{std::move(error_message)}
{
}

inline TaskException::TaskException(std::string error_message)
  : TaskException{0, std::move(error_message)}
{
}

inline const char* TaskException::what() const noexcept { return error_message().c_str(); }

inline std::int32_t TaskException::index() const noexcept { return index_; }

inline const std::string& TaskException::error_message() const noexcept { return error_message_; }

}  // namespace legate
