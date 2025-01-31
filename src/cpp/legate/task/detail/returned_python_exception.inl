/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/task/detail/returned_python_exception.h>

namespace legate::detail {

constexpr ExceptionKind ReturnedPythonException::kind() { return ExceptionKind::PYTHON; }

inline Span<const std::byte> ReturnedPythonException::pickle() const
{
  return bytes_ ? Span<const std::byte>{bytes_->pkl_bytes.get(), bytes_->pkl_size}
                : Span<const std::byte>{nullptr, 0};
}

inline std::string_view ReturnedPythonException::message() const
{
  return bytes_ ? bytes_->msg : std::string_view{""};
}

inline bool ReturnedPythonException::raised() const { return pickle().size(); }

inline std::size_t ReturnedPythonException::legion_buffer_size() const
{
  const auto pkl_size  = pickle().size();
  const auto mess_size = message().size();

  return max_aligned_size_for_type<decltype(kind())>() +
         max_aligned_size_for_type<decltype(pkl_size)>() + pkl_size +
         max_aligned_size_for_type<decltype(mess_size)>() + mess_size;
}

}  // namespace legate::detail
