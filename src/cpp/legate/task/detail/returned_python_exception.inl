/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/returned_python_exception.h>
#include <legate/utilities/detail/align.h>

namespace legate::detail {

constexpr ExceptionKind ReturnedPythonException::kind() { return ExceptionKind::PYTHON; }

inline Span<const std::byte> ReturnedPythonException::pickle() const
{
  return bytes_ ? Span<const std::byte>{bytes_->pkl_bytes.get(), bytes_->pkl_size}
                : Span<const std::byte>{};
}

inline std::string_view ReturnedPythonException::message() const
{
  return bytes_ ? bytes_->msg : std::string_view{""};
}

inline bool ReturnedPythonException::raised() const { return !pickle().empty(); }

inline std::size_t ReturnedPythonException::legion_buffer_size() const
{
  const auto pkl_size  = pickle().size();
  const auto mess_size = message().size();

  return max_aligned_size_for_type<decltype(kind())>() +
         max_aligned_size_for_type<decltype(pkl_size)>() + pkl_size +
         max_aligned_size_for_type<decltype(mess_size)>() + mess_size;
}

}  // namespace legate::detail
