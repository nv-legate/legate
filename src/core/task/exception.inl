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

#include <utility>

namespace legate {

inline TaskException::TaskException(int32_t index, std::string error_message)
  : index_{index}, error_message_{std::move(error_message)}
{
}

inline TaskException::TaskException(std::string error_message)
  : TaskException{0, std::move(error_message)}
{
}

inline const char* TaskException::what() const noexcept { return error_message().c_str(); }

inline int32_t TaskException::index() const noexcept { return index_; }

inline const std::string& TaskException::error_message() const noexcept { return error_message_; }

}  // namespace legate
