/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/reduce.h>

#include <legate.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/operation.h>
#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>
#include <utility>

namespace operation_reduce_test {

using ReduceUnit = DefaultFixture;

namespace {

[[nodiscard]] std::pair<legate::LogicalStore, legate::LogicalStore> make_stores()
{
  const auto shape     = legate::Shape{1};
  const auto data_type = legate::int32();

  auto* const runtime = legate::Runtime::get_runtime();
  return {runtime->create_store(shape, data_type), runtime->create_store(shape, data_type)};
}

[[nodiscard]] legate::detail::Reduce make_reduce(legate::LogicalStore& store,
                                                 legate::LogicalStore& output)
{
  constexpr auto task_id   = legate::LocalTaskID{0};
  constexpr auto unique_id = std::uint64_t{1};
  constexpr auto radix     = std::int32_t{2};
  constexpr auto priority  = std::int32_t{0};

  auto* const runtime = legate::Runtime::get_runtime();
  return legate::detail::Reduce{runtime->impl()->core_library(),
                                store.impl(),
                                output.impl(),
                                task_id,
                                unique_id,
                                radix,
                                priority,
                                runtime->impl()->get_machine()};
}

}  // namespace

TEST_F(ReduceUnit, Kind)
{
  auto [store, output] = make_stores();
  const auto reduce    = make_reduce(store, output);

  ASSERT_EQ(reduce.kind(), legate::detail::Operation::Kind::REDUCE);
}

TEST_F(ReduceUnit, NeedsPartitioning)
{
  auto [store, output] = make_stores();
  const auto reduce    = make_reduce(store, output);

  ASSERT_TRUE(reduce.needs_partitioning());
}

TEST_F(ReduceUnit, NeedsFlushWhenOutputMapped)
{
  auto [store, output] = make_stores();
  const auto reduce    = make_reduce(store, output);

  ASSERT_FALSE(reduce.needs_flush());
  output.impl()->get_region_field()->set_mapped(/*mapped=*/true);
  ASSERT_TRUE(reduce.needs_flush());
  output.impl()->get_region_field()->set_mapped(/*mapped=*/false);
}

TEST_F(ReduceUnit, NeedsFlushWhenStoreMapped)
{
  auto [store, output] = make_stores();
  const auto reduce    = make_reduce(store, output);

  ASSERT_FALSE(reduce.needs_flush());
  store.impl()->get_region_field()->set_mapped(/*mapped=*/true);
  ASSERT_TRUE(reduce.needs_flush());
  store.impl()->get_region_field()->set_mapped(/*mapped=*/false);
}

}  // namespace operation_reduce_test
