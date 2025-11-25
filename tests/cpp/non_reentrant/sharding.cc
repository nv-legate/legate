/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/runtime/detail/shard.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/type_traits.h>

#include <gtest/gtest.h>

#include <type_traits>
#include <utilities/utilities.h>

namespace test_sharding {

namespace {

constexpr std::string_view TOP_LEVEL_TASK_LIBRARY_NAME = "test_sharding_top_level_task";
constexpr std::string_view LINEARIZE_LIBRARY_NAME      = "test_sharding_linearize";

[[nodiscard]] Legion::ShardingFunctor* get_sharding_functor_from_runtime(
  Legion::ShardID sharding_id)
{
  return Legion::Runtime::get_sharding_functor(sharding_id);
}

[[nodiscard]] Legion::ShardingFunctor* get_sharding_functor_from_library(
  std::string_view library_name, std::int64_t local_sharding_id)
{
  auto* const runtime    = legate::Runtime::get_runtime();
  const auto library     = runtime->find_library(library_name);
  const auto sharding_id = library.get_sharding_id(local_sharding_id);

  return get_sharding_functor_from_runtime(sharding_id);
}

void register_sharding_functors(std::string_view library_name, std::int64_t max_shardings)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto created        = false;
  legate::ResourceConfig config;

  config.max_shardings = max_shardings;

  const auto library = runtime->find_or_create_library(library_name, config, nullptr, {}, &created);

  if (created) {
    legate::detail::register_legate_core_sharding_functors(*(library.impl()));
  }
}

[[nodiscard]] Legion::Domain create_1d_domain(std::int32_t lo, std::int32_t hi)
{
  return Legion::Domain{Legion::DomainPoint{Legion::Point<1>{lo}},
                        Legion::DomainPoint{Legion::Point<1>{hi}}};
}

[[nodiscard]] Legion::Domain create_2d_domain(std::int32_t lo_x,
                                              std::int32_t hi_x,
                                              std::int32_t lo_y,
                                              std::int32_t hi_y)
{
  return Legion::Domain{
    Legion::Rect<2>{Legion::Point<2>{lo_x, lo_y}, Legion::Point<2>{hi_x, hi_y}}};
}

class ToplevelTaskShardingTest : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    constexpr auto max_shardings = 10;
    register_sharding_functors(TOP_LEVEL_TASK_LIBRARY_NAME, max_shardings);
  }
};

class LinearizingShardingTest : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    constexpr auto max_shardings = 10;
    register_sharding_functors(LINEARIZE_LIBRARY_NAME, max_shardings);
  }
};

class LegateShardingTest : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    register_legate_sharding_functors();
  }

  void register_legate_sharding_functors()
  {
    auto* const legate_sharding_functor = get_sharding_functor_from_runtime(LEGATE_SHARDING_ID);

    if (legate_sharding_functor == nullptr) {
      legate::detail::create_sharding_functor_using_projection(
        LEGATE_SHARDING_ID, IDENTITY_PROJ_ID, RANGE);

      const auto sharding_id =
        legate::detail::find_sharding_functor_by_projection_functor(IDENTITY_PROJ_ID);

      ASSERT_EQ(sharding_id, LEGATE_SHARDING_ID);
    }
  }

  // Only test the identity projection for now
  static constexpr Legion::ProjectionID IDENTITY_PROJ_ID = 0;
  static constexpr Legion::ShardID LEGATE_SHARDING_ID    = 1000;
  static constexpr legate::mapping::ProcessorRange RANGE = {0, 16, 4};
};

using LegateShardingDeathTest = LegateShardingTest;

}  // namespace

TEST_F(ToplevelTaskShardingTest, IsInvertible)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    TOP_LEVEL_TASK_LIBRARY_NAME,
    legate::detail::to_underlying(legate::detail::CoreShardID::TOPLEVEL_TASK));

  ASSERT_NE(sharding_functor, nullptr);
  ASSERT_FALSE(sharding_functor->is_invertible());
}

TEST_F(ToplevelTaskShardingTest, Shard1DEven)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    TOP_LEVEL_TASK_LIBRARY_NAME,
    legate::detail::to_underlying(legate::detail::CoreShardID::TOPLEVEL_TASK));

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards     = 4;
  constexpr std::size_t total_points     = 16;
  constexpr std::size_t points_per_shard = 4;

  const auto domain = create_1d_domain(0, 15);

  for (std::size_t i = 0; i < total_points; ++i) {
    const auto shard_id_expected = i / points_per_shard;
    const auto shard_id_actual =
      sharding_functor->shard(Legion::DomainPoint{Legion::Point<1>{i}}, domain, total_shards);

    ASSERT_EQ(shard_id_actual, shard_id_expected);
  }
}

TEST_F(ToplevelTaskShardingTest, Shard1DUneven)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    TOP_LEVEL_TASK_LIBRARY_NAME,
    legate::detail::to_underlying(legate::detail::CoreShardID::TOPLEVEL_TASK));

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards     = 4;
  constexpr std::size_t total_points     = 10;
  constexpr std::size_t points_per_shard = 3;

  const auto domain = create_1d_domain(0, 9);

  for (std::size_t i = 0; i < total_points; ++i) {
    const auto shard_id_expected = i / points_per_shard;
    const auto shard_id_actual =
      sharding_functor->shard(Legion::DomainPoint{Legion::Point<1>{i}}, domain, total_shards);

    ASSERT_EQ(shard_id_actual, shard_id_expected);
  }
}

TEST_F(LinearizingShardingTest, IsInvertible)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    LINEARIZE_LIBRARY_NAME, legate::detail::to_underlying(legate::detail::CoreShardID::LINEARIZE));

  ASSERT_NE(sharding_functor, nullptr);
  ASSERT_TRUE(sharding_functor->is_invertible());
}

TEST_F(LinearizingShardingTest, Shard2DEven)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    LINEARIZE_LIBRARY_NAME, legate::detail::to_underlying(legate::detail::CoreShardID::LINEARIZE));

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards   = 4;
  constexpr std::size_t points_per_dim = 4;

  const auto domain = create_2d_domain(0, 3, 0, 3);

  for (std::size_t i = 0; i < points_per_dim; ++i) {
    for (std::size_t j = 0; j < points_per_dim; ++j) {
      const auto shard_id_expected = i;
      const auto shard_id_actual =
        sharding_functor->shard(Legion::DomainPoint{Legion::Point<2>{i, j}}, domain, total_shards);

      ASSERT_EQ(shard_id_actual, shard_id_expected);
    }
  }
}

TEST_F(LinearizingShardingTest, Shard2DUneven)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    LINEARIZE_LIBRARY_NAME, legate::detail::to_underlying(legate::detail::CoreShardID::LINEARIZE));

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards = 4;

  const auto domain = create_2d_domain(0, 4, 0, 2);

  const auto check_shard = [&](const Legion::Point<2>& point, std::size_t expected_shard_id) {
    const auto shard_id = sharding_functor->shard(Legion::DomainPoint{point}, domain, total_shards);

    ASSERT_EQ(shard_id, expected_shard_id);
  };

  // Verify start point of shard 0
  check_shard({0, 0}, 0);

  // Verify start point of shard 1
  check_shard({1, 1}, 1);

  // Verify start point of shard 2
  check_shard({2, 2}, 2);

  // Verify start point of shard 3
  check_shard({4, 0}, 3);
}

TEST_F(LinearizingShardingTest, Invert2DEven)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    LINEARIZE_LIBRARY_NAME, legate::detail::to_underlying(legate::detail::CoreShardID::LINEARIZE));

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards     = 4;
  constexpr std::size_t points_per_dim   = 4;
  constexpr std::size_t points_per_shard = 4;

  const auto domain = create_2d_domain(0, 3, 0, 3);

  for (std::size_t i = 0; i < points_per_dim; ++i) {
    std::vector<Legion::DomainPoint> points;
    std::vector<Legion::DomainPoint> expected_points;

    sharding_functor->invert(i, domain, domain, total_shards, points);
    ASSERT_EQ(points.size(), points_per_shard);

    for (std::size_t j = 0; j < points_per_dim; ++j) {
      expected_points.emplace_back(Legion::Point<2>{i, j});

      // Check the shard value generated by inverted point
      ASSERT_EQ(sharding_functor->shard(points[j], domain, total_shards), i);
    }

    // Check the point value
    ASSERT_THAT(points, ::testing::ContainerEq(expected_points));
  }
}

TEST_F(LinearizingShardingTest, Invert2DUneven)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    LINEARIZE_LIBRARY_NAME, legate::detail::to_underlying(legate::detail::CoreShardID::LINEARIZE));

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards = 4;

  const auto domain = create_2d_domain(0, 4, 0, 2);

  auto check_invert =
    [&](std::size_t shard_id, const Legion::Point<2>& first_point, std::size_t expected_size) {
      std::vector<Legion::DomainPoint> points;

      sharding_functor->invert(shard_id, domain, domain, total_shards, points);
      ASSERT_EQ(points.size(), expected_size);
      ASSERT_EQ(points[0], (Legion::DomainPoint{first_point}));
      ASSERT_EQ(sharding_functor->shard(points[0], domain, total_shards), shard_id);
    };

  // Verify invert for shard 0
  check_invert(0, {0, 0}, 4);

  // Verify invert for shard 1
  check_invert(1, {1, 1}, 4);

  // Verify invert for shard 2
  check_invert(2, {2, 2}, 4);

  // Verify invert for shard 3
  check_invert(3, {4, 0}, 3);
}

TEST_F(LinearizingShardingTest, Invert2DOutOfRange)
{
  auto* const sharding_functor = get_sharding_functor_from_library(
    LINEARIZE_LIBRARY_NAME, legate::detail::to_underlying(legate::detail::CoreShardID::LINEARIZE));

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards = 4;
  constexpr std::size_t shard_id     = 4;

  const auto domain = create_2d_domain(0, 3, 0, 3);
  std::vector<Legion::DomainPoint> points;

  sharding_functor->invert(shard_id, domain, domain, total_shards, points);

  ASSERT_EQ(points.size(), 0);
}

TEST_F(LegateShardingTest, CreateWithInvalidProjection)
{
  constexpr Legion::ProjectionID invalid_proj_id = 10000;

  ASSERT_THAT(
    [&] {
      legate::detail::create_sharding_functor_using_projection(
        LEGATE_SHARDING_ID, invalid_proj_id, RANGE);
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Failed to find projection functor of id 1000")));
}

TEST_F(LegateShardingTest, IsInvertible)
{
  auto* const sharding_functor = get_sharding_functor_from_runtime(LEGATE_SHARDING_ID);

  ASSERT_NE(sharding_functor, nullptr);
  ASSERT_FALSE(sharding_functor->is_invertible());
}

TEST_F(LegateShardingTest, Shard1DEven)
{
  auto* const sharding_functor = get_sharding_functor_from_runtime(LEGATE_SHARDING_ID);

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards = 4;
  constexpr std::size_t total_points = 16;

  const auto domain = create_1d_domain(0, 15);

  for (std::size_t i = 0; i < total_points; ++i) {
    const auto shard_id_expected = i / RANGE.per_node_count;
    const auto shard_id_actual =
      sharding_functor->shard(Legion::DomainPoint{Legion::Point<1>{i}}, domain, total_shards);

    ASSERT_EQ(shard_id_actual, shard_id_expected);
  }
}

TEST_F(LegateShardingTest, Shard1DUneven)
{
  auto* const sharding_functor = get_sharding_functor_from_runtime(LEGATE_SHARDING_ID);

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards = 4;

  const auto domain = create_1d_domain(0, 9);

  // Only check the start point and end point for uneven case because it's hard to verify all points
  // Verify start point of shard 0 (first shard)
  {
    const auto shard_id =
      sharding_functor->shard(Legion::DomainPoint{Legion::Point<1>{0}}, domain, total_shards);

    ASSERT_EQ(shard_id, 0);
  }

  // Verify end point of shard 3 (last shard)
  {
    const auto shard_id =
      sharding_functor->shard(Legion::DomainPoint{Legion::Point<1>{9}}, domain, total_shards);

    ASSERT_EQ(shard_id, 3);
  }
}

TEST_F(LegateShardingTest, Shard2DEven)
{
  auto* const sharding_functor = get_sharding_functor_from_runtime(LEGATE_SHARDING_ID);

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards   = 4;
  constexpr std::size_t points_per_dim = 4;

  const auto domain = create_2d_domain(0, 3, 0, 3);

  for (std::size_t i = 0; i < points_per_dim; ++i) {
    for (std::size_t j = 0; j < points_per_dim; ++j) {
      const auto shard_id_expected = i;
      const auto shard_id_actual =
        sharding_functor->shard(Legion::DomainPoint{Legion::Point<2>{i, j}}, domain, total_shards);

      ASSERT_EQ(shard_id_actual, shard_id_expected);
    }
  }
}

TEST_F(LegateShardingTest, Shard2DUneven)
{
  auto* const sharding_functor = get_sharding_functor_from_runtime(LEGATE_SHARDING_ID);

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards = 4;

  const auto domain = create_2d_domain(0, 4, 0, 2);

  // Only check the start point and end point for uneven case because it's hard to verify all points
  // Verify start point of shard 0 (first shard)
  {
    const auto shard_id =
      sharding_functor->shard(Legion::DomainPoint{Legion::Point<2>{0, 0}}, domain, total_shards);

    ASSERT_EQ(shard_id, 0);
  }

  // Verify end point of shard 3 (last shard)
  {
    const auto shard_id =
      sharding_functor->shard(Legion::DomainPoint{Legion::Point<2>{4, 2}}, domain, total_shards);

    ASSERT_EQ(shard_id, 3);
  }
}

TEST_F(LegateShardingDeathTest, ShardInvalidRange)
{
  // Skip this test if LEGATE_USE_DEBUG is not defined
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skipping test due to verifying logic in LEGATE_ASSERT()";
  }

  auto* const sharding_functor = get_sharding_functor_from_runtime(LEGATE_SHARDING_ID);

  ASSERT_NE(sharding_functor, nullptr);

  constexpr std::size_t total_shards = 3;
  constexpr std::size_t total_points = 16;

  const auto domain = create_1d_domain(0, 15);

  ASSERT_EXIT(static_cast<void>(sharding_functor->shard(
                Legion::DomainPoint{Legion::Point<1>{total_points}}, domain, total_shards)),
              ::testing::KilledBySignal{SIGABRT},
              "shard_id < total_shards");
}

TEST_F(LegateShardingDeathTest, FindByInvalidProjection)
{
  constexpr Legion::ProjectionID invalid_proj_id = 10000;

  ASSERT_EXIT(
    static_cast<void>(legate::detail::find_sharding_functor_by_projection_functor(invalid_proj_id)),
    ::testing::KilledBySignal{SIGABRT},
    "it != functor_id_table.end\\(\\)");
}

TEST_F(LegateShardingDeathTest, CreateWithDuplicateID)
{
  // Create a sharding functor with the same ID, the first creation is inside the SetUp() function
  ASSERT_EXIT(legate::detail::create_sharding_functor_using_projection(
                LEGATE_SHARDING_ID, IDENTITY_PROJ_ID, RANGE),
              ::testing::KilledBySignal{SIGABRT},
              "");
}

}  // namespace test_sharding
