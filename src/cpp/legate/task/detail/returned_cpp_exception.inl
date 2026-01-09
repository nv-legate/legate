/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/returned_cpp_exception.h>

namespace legate::detail {

inline ReturnedCppException::ReturnedCppException(std::int32_t index, std::string error_message)
  : index_{index}, message_{std::move(error_message)}
{
}

inline std::int32_t ReturnedCppException::index() const { return index_; }

inline ZStringView ReturnedCppException::message() const { return message_; }

inline std::uint64_t ReturnedCppException::size() const
{
  return static_cast<std::uint64_t>(message().size());
}

inline bool ReturnedCppException::raised() const { return !message().empty(); }

// NOLINTNEXTLINE(readability-redundant-inline-specifier)
inline constexpr ExceptionKind ReturnedCppException::kind() { return ExceptionKind::CPP; }

}  // namespace legate::detail
