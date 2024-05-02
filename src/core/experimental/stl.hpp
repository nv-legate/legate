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

#include "core/experimental/stl/detail/stlfwd.hpp"
//
#include "core/experimental/stl/detail/fill.hpp"
#include "core/experimental/stl/detail/for_each.hpp"
#include "core/experimental/stl/detail/launch_task.hpp"
#include "core/experimental/stl/detail/reduce.hpp"
#include "core/experimental/stl/detail/registrar.hpp"
#include "core/experimental/stl/detail/slice.hpp"
#include "core/experimental/stl/detail/transform.hpp"
#include "core/experimental/stl/detail/transform_reduce.hpp"

// Include this last:
#include "core/experimental/stl/detail/prefix.hpp"

namespace legate::experimental::stl {
}

#include "core/experimental/stl/detail/suffix.hpp"
