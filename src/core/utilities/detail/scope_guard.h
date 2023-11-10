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

namespace legate::detail {

template <typename T>
class ScopeGuard {
 public:
  explicit ScopeGuard(T&& fn);

  ~ScopeGuard();

 private:
  T fn_;
};

template <typename T>
[[nodiscard]] ScopeGuard<T> make_scope_guard(T&& fn);

}  // namespace legate::detail

#include "core/utilities/detail/scope_guard.inl"
