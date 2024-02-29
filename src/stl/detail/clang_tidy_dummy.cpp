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

#include "stl/detail/stlfwd.hpp"
#include "stl/stl.hpp"
//
#include "stl/detail/fill.hpp"
#include "stl/detail/for_each.hpp"
#include "stl/detail/launch_task.hpp"
#include "stl/detail/reduce.hpp"
#include "stl/detail/registrar.hpp"
#include "stl/detail/slice.hpp"
#include "stl/detail/transform.hpp"
#include "stl/detail/transform_reduce.hpp"

// Include this last:
#include "stl/detail/prefix.hpp"

// This file exists only to add a cpp file that includes the headers, so that clang-tidy will
// check them for us
namespace legate::stl::detail::clang_tidy_dummy {

void internal_private_do_not_call_or_you_will_be_fired() {}

}  // namespace legate::stl::detail::clang_tidy_dummy

#include "stl/detail/suffix.hpp"
