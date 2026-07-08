/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/scatter.h>

#include <legate.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/operation.h>
#include <legate/runtime/detail/runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <utilities/utilities.h>
#include <utility>

namespace operation_scatter_test {

using ScatterUnit = DefaultFixture;

namespace {

struct ScatterTestData {
  legate::LogicalStore target;
  legate::LogicalStore target_indirect;
  legate::LogicalStore source;
  std::unique_ptr<legate::detail::Scatter> scatter;
};

[[nodiscard]] ScatterTestData make_scatter()
{
  constexpr auto unique_id = std::uint64_t{1};
  constexpr auto priority  = std::int32_t{0};
  const auto shape         = legate::Shape{1};
  const auto data_type     = legate::int64();

  auto* const runtime  = legate::Runtime::get_runtime();
  auto target          = runtime->create_store(shape, data_type);
  auto target_indirect = runtime->create_store(shape, legate::point_type(1));
  auto source          = runtime->create_store(shape, data_type);
  auto scatter         = std::make_unique<legate::detail::Scatter>(target.impl(),
                                                                   target_indirect.impl(),
                                                                   source.impl(),
                                                                   unique_id,
                                                                   priority,
                                                                   runtime->impl()->get_machine(),
                                                                   std::nullopt);

  return {std::move(target), std::move(target_indirect), std::move(source), std::move(scatter)};
}

}  // namespace

TEST_F(ScatterUnit, Kind)
{
  const auto data = make_scatter();

  ASSERT_EQ(data.scatter->kind(), legate::detail::Operation::Kind::SCATTER);
}

TEST_F(ScatterUnit, NeedsFlush)
{
  auto data = make_scatter();

  ASSERT_FALSE(data.scatter->needs_flush());

  data.target.impl()->get_region_field()->set_mapped(/*mapped=*/true);
  ASSERT_TRUE(data.scatter->needs_flush());
  data.target.impl()->get_region_field()->set_mapped(/*mapped=*/false);

  data.source.impl()->get_region_field()->set_mapped(/*mapped=*/true);
  ASSERT_TRUE(data.scatter->needs_flush());
  data.source.impl()->get_region_field()->set_mapped(/*mapped=*/false);

  data.target_indirect.impl()->get_region_field()->set_mapped(/*mapped=*/true);
  ASSERT_TRUE(data.scatter->needs_flush());
  data.target_indirect.impl()->get_region_field()->set_mapped(/*mapped=*/false);
}

}  // namespace operation_scatter_test
