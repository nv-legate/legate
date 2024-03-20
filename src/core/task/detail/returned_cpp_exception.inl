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

#include "core/task/detail/returned_cpp_exception.h"

namespace legate::detail {

inline ReturnedCppException::ReturnedCppException(std::int32_t index, std::string error_message)
  : index_{index}, message_{std::move(error_message)}
{
}

inline std::int32_t ReturnedCppException::index() const { return index_; }

inline std::uint64_t ReturnedCppException::size() const
{
  return static_cast<std::uint64_t>(message_.size());
}

// NOLINTNEXTLINE(readability-redundant-inline-specifier)
inline bool ReturnedCppException::raised() const { return !message_.empty(); }

// NOLINTNEXTLINE(readability-redundant-inline-specifier)
inline constexpr ExceptionKind ReturnedCppException::kind() { return ExceptionKind::CPP; }

}  // namespace legate::detail
