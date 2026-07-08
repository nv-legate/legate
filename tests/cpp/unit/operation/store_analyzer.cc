/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/operation/detail/store_analyzer.h>

#include <legate.h>

#include <legate/data/detail/logical_region_field.h>
#include <legate/data/detail/logical_store.h>
#include <legate/operation/detail/store_projection.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utilities/utilities.h>

namespace operation_store_analyzer_test {

using ProjectionSetUnit       = ::testing::Test;
using RequirementAnalyzerUnit = DefaultFixture;
using FutureAnalyzerUnit      = ::testing::Test;

namespace {

constexpr auto STRICT_INTERFERENCE_CHECKS  = false;
constexpr auto RELAXED_INTERFERENCE_CHECKS = true;
constexpr auto WRITE_ONLY_WITH_DISCARD =
  static_cast<Legion::PrivilegeMode>(LEGION_WRITE_ONLY | LEGION_DISCARD_OUTPUT_MASK);

[[nodiscard]] legate::detail::StoreProjection make_projection(Legion::ProjectionID proj_id)
{
  auto projection    = legate::detail::StoreProjection{};
  projection.proj_id = proj_id;
  return projection;
}

}  // namespace

TEST_F(ProjectionSetUnit, InsertThrowsOnConflictingWriteProjections)
{
  auto projections  = legate::detail::ProjectionSet{};
  const auto first  = make_projection(Legion::ProjectionID{0});
  const auto second = make_projection(Legion::ProjectionID{1});

  projections.insert(LEGION_READ_ONLY, first, STRICT_INTERFERENCE_CHECKS);

  ASSERT_THAT([&] { projections.insert(LEGION_WRITE_ONLY, second, STRICT_INTERFERENCE_CHECKS); },
              ::testing::Throws<legate::detail::InterferingStoreError>());
}

TEST_F(ProjectionSetUnit, ExistingNoAccessSkipsPrivilegePromotion)
{
  auto projections  = legate::detail::ProjectionSet{};
  const auto first  = make_projection(Legion::ProjectionID{0});
  const auto second = make_projection(Legion::ProjectionID{1});

  projections.insert(LEGION_NO_ACCESS, first, STRICT_INTERFERENCE_CHECKS);
  projections.insert(LEGION_WRITE_ONLY, second, STRICT_INTERFERENCE_CHECKS);

  ASSERT_EQ(projections.privilege(), LEGION_NO_ACCESS);
  ASSERT_EQ(projections.store_projs().size(), std::size_t{2});
}

TEST_F(ProjectionSetUnit, NewNoAccessSkipsPrivilegePromotion)
{
  auto projections  = legate::detail::ProjectionSet{};
  const auto first  = make_projection(Legion::ProjectionID{0});
  const auto second = make_projection(Legion::ProjectionID{1});

  projections.insert(LEGION_WRITE_ONLY, first, STRICT_INTERFERENCE_CHECKS);
  projections.insert(LEGION_NO_ACCESS, second, RELAXED_INTERFERENCE_CHECKS);

  ASSERT_EQ(projections.privilege(), LEGION_WRITE_ONLY);
  ASSERT_EQ(projections.store_projs().size(), std::size_t{2});
}

TEST_F(RequirementAnalyzerUnit, PopulateWriteOnlyStreamingDiscardSuppressesWarnings)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto store          = runtime->create_store(legate::Shape{1}, legate::int64());
  const auto& field   = store.impl()->get_region_field();
  auto analyzer       = legate::detail::RequirementAnalyzer{};

  analyzer.insert(
    field->region(), field->field_id(), WRITE_ONLY_WITH_DISCARD, legate::detail::StoreProjection{});
  analyzer.analyze_requirements();

  auto launcher = Legion::TaskLauncher{};

  analyzer.populate_launcher(launcher);

  ASSERT_EQ(launcher.region_requirements.size(), std::size_t{1});
  ASSERT_NE(launcher.region_requirements.front().flags & LEGION_SUPPRESS_WARNINGS_FLAG, 0);
}

TEST_F(FutureAnalyzerUnit, InsertFutureAssignsIndex)
{
  auto analyzer                 = legate::detail::FutureAnalyzer{};
  auto future                   = Legion::Future{};
  auto launcher                 = Legion::TaskLauncher{};
  constexpr auto expected_index = std::int32_t{0};

  analyzer.insert(future);
  analyzer.analyze_futures();
  analyzer.populate_launcher(launcher);

  ASSERT_EQ(analyzer.get_index(future), expected_index);
  ASSERT_EQ(launcher.futures.size(), std::size_t{1});
  ASSERT_EQ(launcher.futures.front(), future);
}

}  // namespace operation_store_analyzer_test
