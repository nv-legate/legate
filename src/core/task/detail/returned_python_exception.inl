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

#include "core/task/detail/returned_python_exception.h"

namespace legate::detail {

inline ReturnedPythonException::ReturnedPythonException(const void* buf, std::size_t len)
  : ReturnedPythonException{Span<const void>{buf, len}}
{
}

inline ReturnedPythonException::ReturnedPythonException(Span<const void> span)
  : ReturnedPythonException{span, InternalSharedPtr<char[]>{new char[span.size()]}}
{
}

inline ReturnedPythonException::ReturnedPythonException(Span<const void> span,
                                                        InternalSharedPtr<char[]> mem)
  : size_{span.size()}, pickle_bytes_{std::move(mem)}
{
  if (size()) {
    std::memcpy(pickle_bytes_.get(), span.ptr(), size());
  }
}

inline ReturnedPythonException::ReturnedPythonException(ReturnedPythonException&& other) noexcept
  : size_{std::exchange(other.size_, 0)}, pickle_bytes_{std::move(other.pickle_bytes_)}
{
}

inline ReturnedPythonException& ReturnedPythonException::operator=(
  ReturnedPythonException&& other) noexcept
{
  if (this != &other) {
    size_         = std::exchange(other.size_, 0);
    pickle_bytes_ = std::move(other.pickle_bytes_);
  }
  return *this;
}

// NOLINTNEXTLINE(readability-redundant-inline-specifier)
inline constexpr ExceptionKind ReturnedPythonException::kind() { return ExceptionKind::PYTHON; }

inline const void* ReturnedPythonException::data() const { return pickle_bytes_.get(); }

inline std::uint64_t ReturnedPythonException::size() const { return size_; }

inline bool ReturnedPythonException::raised() const { return data(); }

inline std::size_t ReturnedPythonException::legion_buffer_size() const
{
  return max_aligned_size_for_type<decltype(kind())>() +
         max_aligned_size_for_type<decltype(size())>() + size();
}

}  // namespace legate::detail
