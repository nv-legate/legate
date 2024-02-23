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

#include "store.hpp"

// Include this last
#include "prefix.hpp"

namespace legate::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Range>              //
  requires(logical_store_like<Range>)  //
void fill(Range&& output, value_type_of_t<Range> val)
{
  auto store                    = get_logical_store(output);
  observer_ptr<Runtime> runtime = legate::Runtime::get_runtime();
  runtime->issue_fill(store, Scalar(val));
}

}  // namespace legate::stl

#include "suffix.hpp"
