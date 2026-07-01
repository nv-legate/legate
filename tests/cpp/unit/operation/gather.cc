/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/gather.h>

#include <legate.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utilities/utilities.h>
#include <utility>

namespace operation_gather_test {

using GatherUnit = DefaultFixture;

namespace {

struct GatherTestData {
  legate::LogicalStore target;
  legate::LogicalStore source;
  legate::LogicalStore source_indirect;
  std::unique_ptr<legate::detail::Gather> gather;
};

[[nodiscard]] GatherTestData make_gather()
{
  constexpr auto unique_id = std::uint64_t{1};
  constexpr auto priority  = std::int32_t{0};
  const auto shape         = legate::Shape{1};
  const auto data_type     = legate::int64();

  auto* const runtime  = legate::Runtime::get_runtime();
  auto target          = runtime->create_store(shape, data_type);
  auto source          = runtime->create_store(shape, data_type);
  auto source_indirect = runtime->create_store(shape, legate::point_type(1));
  auto gather          = std::make_unique<legate::detail::Gather>(target.impl(),
                                                                  source.impl(),
                                                                  source_indirect.impl(),
                                                                  unique_id,
                                                                  priority,
                                                                  runtime->impl()->get_machine(),
                                                                  std::nullopt);

  return {std::move(target), std::move(source), std::move(source_indirect), std::move(gather)};
}

}  // namespace

TEST_F(GatherUnit, Kind)
{
  const auto data = make_gather();

  ASSERT_EQ(data.gather->kind(), legate::detail::Operation::Kind::GATHER);
}

TEST_F(GatherUnit, NeedsFlush)
{
  auto data = make_gather();

  ASSERT_FALSE(data.gather->needs_flush());

  data.target.impl()->get_region_field()->set_mapped(/*mapped=*/true);
  ASSERT_TRUE(data.gather->needs_flush());
  data.target.impl()->get_region_field()->set_mapped(/*mapped=*/false);

  data.source.impl()->get_region_field()->set_mapped(/*mapped=*/true);
  ASSERT_TRUE(data.gather->needs_flush());
  data.source.impl()->get_region_field()->set_mapped(/*mapped=*/false);

  data.source_indirect.impl()->get_region_field()->set_mapped(/*mapped=*/true);
  ASSERT_TRUE(data.gather->needs_flush());
  data.source_indirect.impl()->get_region_field()->set_mapped(/*mapped=*/false);
}

}  // namespace operation_gather_test
