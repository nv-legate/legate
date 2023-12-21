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

#include "core/mapping/store.h"

namespace legate::mapping {

template <int32_t DIM>
Rect<DIM> Store::shape() const
{
  static_assert(DIM <= LEGATE_MAX_DIM);
  return Rect<DIM>{domain()};
}

inline const detail::Store* Store::impl() const noexcept { return impl_; }

inline Store::Store(const detail::Store* impl) noexcept : impl_{impl} {}

}  // namespace legate::mapping
