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

#include "core/partitioning/constraint.h"

namespace legate {

inline Variable::Variable(const detail::Variable* impl) : impl_{impl} {}

inline const detail::Variable* Variable::impl() const { return impl_; }

// ==========================================================================================

inline const SharedPtr<detail::Constraint>& Constraint::impl() const { return impl_; }

}  // namespace legate
