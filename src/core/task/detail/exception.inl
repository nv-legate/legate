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

#include "core/task/detail/exception.h"

#include <utility>

namespace legate::detail {

inline PythonTaskException::PythonTaskException(std::uint64_t size, SharedPtr<const char[]> buf)
  : TaskException{"Python exception"}, size_{size}, bytes_{std::move(buf)}
{
}

inline const char* PythonTaskException::data() const noexcept { return bytes_.get(); }

inline std::uint64_t PythonTaskException::size() const noexcept { return size_; }

}  // namespace legate::detail
