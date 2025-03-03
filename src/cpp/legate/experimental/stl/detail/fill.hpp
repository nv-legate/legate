/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/experimental/stl/detail/store.hpp>

// Include this last
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

////////////////////////////////////////////////////////////////////////////////////////////////////
/**
 * Fills the given range with the specified value.
 *
 * This function fills the elements in the range [begin, end) with the specified value.
 * The range must be a logical store-like object, meaning it supports the necessary
 * operations for storing values. The value to be filled is specified by the `val`
 * parameter.
 *
 * @param output The range to be filled.
 * @param val The value to fill the range with.
 *
 * @par Example:
 * @snippet{trimleft} experimental/stl/fill.cc fill example
 *
 * @ingroup stl-algorithms
 */
template <typename Range>              //
  requires(logical_store_like<Range>)  //
void fill(Range&& output, value_type_of_t<Range> val)
{
  auto store                    = get_logical_store(std::forward<Range>(output));
  observer_ptr<Runtime> runtime = legate::Runtime::get_runtime();
  runtime->issue_fill(std::move(store), Scalar{std::move(val)});
}

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
