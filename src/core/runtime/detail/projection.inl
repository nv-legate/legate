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

#include "core/runtime/detail/projection.h"

namespace legate::detail {

inline LegateProjectionFunctor::LegateProjectionFunctor(Legion::Runtime* lg_runtime)
  : ProjectionFunctor{lg_runtime}
{
}

inline bool LegateProjectionFunctor::is_functional() const { return true; }

inline bool LegateProjectionFunctor::is_exclusive() const { return true; }

inline unsigned LegateProjectionFunctor::get_depth() const { return 0; }

}  // namespace legate::detail
