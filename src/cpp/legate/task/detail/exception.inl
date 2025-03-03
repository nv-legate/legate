/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/exception.h>

#include <utility>

namespace legate::detail {

inline PythonTaskException::PythonTaskException(std::uint64_t size,
                                                SharedPtr<const std::byte[]> buf)
  : TaskException{"Python exception"}, size_{size}, bytes_{std::move(buf)}
{
}

inline const std::byte* PythonTaskException::data() const noexcept { return bytes_.get(); }

inline std::uint64_t PythonTaskException::size() const noexcept { return size_; }

}  // namespace legate::detail
