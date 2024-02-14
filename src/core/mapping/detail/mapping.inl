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

#include "core/mapping/detail/mapping.h"

#include <utility>

namespace legate::mapping::detail {

inline DimOrdering::DimOrdering(Kind _kind) : kind{_kind} {}

inline DimOrdering::DimOrdering(std::vector<std::int32_t> _dims)
  : kind{Kind::CUSTOM}, dims{std::move(_dims)}
{
}

inline bool DimOrdering::operator==(const DimOrdering& other) const
{
  return kind == other.kind && dims == other.dims;
}

}  // namespace legate::mapping::detail
