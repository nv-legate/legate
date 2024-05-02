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

#include "core/experimental/stl/detail/stlfwd.hpp"
#include "core/experimental/stl.hpp"
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

// This file exists only to add a cpp file that includes the headers, so that clang-tidy will
// check them for us
namespace legate::experimental::stl::detail::clang_tidy_dummy {

void internal_private_do_not_call_or_you_will_be_fired() {}

}  // namespace legate::experimental::stl::detail::clang_tidy_dummy

#include "core/experimental/stl/detail/suffix.hpp"
