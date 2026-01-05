/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/non_invertible_transformation.h>

namespace legate::detail {

NonInvertibleTransformation::NonInvertibleTransformation()
  : runtime_error{"Non-invertible transformation"}
{
}

}  // namespace legate::detail
