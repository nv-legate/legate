/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/constraint.h>

namespace legate {

inline Variable::Variable(const detail::Variable* impl) : impl_{impl} {}

inline const detail::Variable* Variable::impl() const { return impl_; }

// ==========================================================================================

inline const SharedPtr<detail::Constraint>& Constraint::impl() const { return impl_; }

}  // namespace legate
