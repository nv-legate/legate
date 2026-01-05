/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/shape.h>

namespace legate {

inline Shape::Shape(InternalSharedPtr<detail::Shape> impl) : impl_{std::move(impl)} {}

inline const SharedPtr<detail::Shape>& Shape::impl() const { return impl_; }

}  // namespace legate
