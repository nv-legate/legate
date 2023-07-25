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

// Useful for IDEs
#include "core/mapping/operation.h"

namespace legate::mapping {

template <int32_t DIM>
Legion::Rect<DIM> RegionField::shape(Legion::Mapping::MapperRuntime* runtime,
                                     const Legion::Mapping::MapperContext context) const
{
  return Legion::Rect<DIM>(domain(runtime, context));
}

template <int32_t DIM>
Legion::Rect<DIM> FutureWrapper::shape() const
{
  return Legion::Rect<DIM>(domain());
}

template <int32_t DIM>
Legion::Rect<DIM> Store::shape() const
{
  return Legion::Rect<DIM>(domain());
}

}  // namespace legate::mapping
