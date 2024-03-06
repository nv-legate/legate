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

#include "core/task/detail/returned_exception.h"

namespace legate::detail {

#if LEGATE_CPP_VERSION < 26
template <typename T>
decltype(auto) ReturnedException::visit(T&& fn)
{
  return std::visit(std::forward<T>(fn), *this);
}

template <typename T>
decltype(auto) ReturnedException::visit(T&& fn) const
{
  return std::visit(std::forward<T>(fn), *this);
}
#endif

// ==========================================================================================

}  // namespace legate::detail
