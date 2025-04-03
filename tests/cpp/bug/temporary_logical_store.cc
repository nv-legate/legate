/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace temporary_logical_store_test {

class TemporaryLogicalStoreUnit : public DefaultFixture {};

TEST_F(TemporaryLogicalStoreUnit, Store)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto shape    = legate::Shape{1, 2, 3};
  const auto ty       = legate::int32();

  const auto phys_store   = runtime->create_store(shape, ty).get_physical_store();
  const auto inline_alloc = phys_store.get_inline_allocation();

  const auto phys_store_2   = runtime->create_store(shape, ty).get_physical_store();
  const auto inline_alloc_2 = phys_store_2.get_inline_allocation();

  static_cast<void>(phys_store);
  static_cast<void>(inline_alloc);
  static_cast<void>(phys_store_2);
  static_cast<void>(inline_alloc_2);
  // For this bug to be tested, we should get here without Legion aborting the program:
  //
  //  LEGION ERROR: Attempted an inline mapping of region (3,3,3) that conflicts with previous
  //  inline mapping in task Legate Core Toplevel Task (ID 1) that would ultimately result in
  //  deadlock.  Instead you receive this error message. (from file
  //  /path/to/_deps/legion-src/runtime/legion/legion_context.cc:7312)
}

TEST_F(TemporaryLogicalStoreUnit, Array)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto shape    = legate::Shape{1, 2, 3};
  const auto ty       = legate::int32();

  const auto phys_array   = runtime->create_array(shape, ty).get_physical_array();
  const auto inline_alloc = phys_array.data().get_inline_allocation();

  const auto phys_array_2   = runtime->create_array(shape, ty).get_physical_array();
  const auto inline_alloc_2 = phys_array_2.data().get_inline_allocation();

  static_cast<void>(phys_array);
  static_cast<void>(inline_alloc);
  static_cast<void>(phys_array_2);
  static_cast<void>(inline_alloc_2);
  // For this bug to be tested, we should get here without Legion aborting the program:
  //
  //  LEGION ERROR: Attempted an inline mapping of region (3,3,3) that conflicts with previous
  //  inline mapping in task Legate Core Toplevel Task (ID 1) that would ultimately result in
  //  deadlock.  Instead you receive this error message. (from file
  //  /path/to/_deps/legion-src/runtime/legion/legion_context.cc:7312)
}

TEST_F(TemporaryLogicalStoreUnit, Mixed)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto shape    = legate::Shape{1, 2, 3};
  const auto ty       = legate::int32();

  const auto phys_array   = runtime->create_array(shape, ty).get_physical_array();
  const auto inline_alloc = phys_array.data().get_inline_allocation();

  const auto phys_store     = runtime->create_store(shape, ty).get_physical_store();
  const auto inline_alloc_2 = phys_store.get_inline_allocation();

  static_cast<void>(phys_array);
  static_cast<void>(inline_alloc);
  static_cast<void>(phys_store);
  static_cast<void>(inline_alloc_2);
  // For this bug to be tested, we should get here without Legion aborting the program:
  //
  //  LEGION ERROR: Attempted an inline mapping of region (3,3,3) that conflicts with previous
  //  inline mapping in task Legate Core Toplevel Task (ID 1) that would ultimately result in
  //  deadlock.  Instead you receive this error message. (from file
  //  /path/to/_deps/legion-src/runtime/legion/legion_context.cc:7312)
}

}  // namespace temporary_logical_store_test
