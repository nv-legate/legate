/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/shape.h>

#include <legate.h>

#include <legate/runtime/detail/runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <utilities/utilities.h>

namespace shape_test {

namespace {

class UnboundStoreTask : public legate::LegateTask<UnboundStoreTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

/*static*/ void UnboundStoreTask::cpu_variant(legate::TaskContext context)
{
  auto store1 = context.output(0).data();
  auto store2 = context.output(1).data();
  auto store3 = context.output(2).data();

  store1.bind_empty_data();
  store2.bind_empty_data();
  static_cast<void>(store3.create_output_buffer<std::uint64_t, LEGATE_MAX_DIM>(
    legate::Point<LEGATE_MAX_DIM>{1}, /*bind_buffer=*/true));
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_bound_shape";

  static void registration_callback(legate::Library library)
  {
    UnboundStoreTask::register_variants(library);
  }
};

class ShapeUnit : public RegisterOnceFixture<Config> {};

class CreateShapeTest
  : public ShapeUnit,
    public ::testing::WithParamInterface<
      std::tuple<legate::Shape, std::uint32_t, std::size_t, legate::tuple<std::uint64_t>>> {};

class EmptyShapeTest : public ShapeUnit, public ::testing::WithParamInterface<legate::Shape> {};

INSTANTIATE_TEST_SUITE_P(
  ShapeUnit,
  CreateShapeTest,
  ::testing::Values(
    std::make_tuple(legate::Shape{1, 3, 5}, 3, 15, legate::tuple<std::uint64_t>{1, 3, 5}),
    std::make_tuple(
      legate::Shape{std::vector<std::uint64_t>{2, 4}}, 2, 8, legate::tuple<std::uint64_t>{2, 4})));

INSTANTIATE_TEST_SUITE_P(ShapeUnit,
                         EmptyShapeTest,
                         ::testing::Values(legate::Shape{},
                                           legate::Shape{std::vector<std::uint64_t>{}}));

}  // namespace

TEST_F(ShapeUnit, Unbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int16(), /*dim=*/1);
  auto shape   = store.shape();

  ASSERT_EQ(shape.ndim(), store.dim());
  ASSERT_THAT([&] { static_cast<void>(shape.volume()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_THAT([&] { static_cast<void>(shape.extents()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
}

TEST_F(ShapeUnit, Bound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{1, 3, 5}, legate::uint32());
  auto shape   = store.shape();

  ASSERT_EQ(shape.ndim(), store.dim());
  ASSERT_EQ(shape.volume(), store.volume());
  ASSERT_EQ(shape.extents(), (legate::tuple<std::uint64_t>{1, 3, 5}));
  ASSERT_EQ(shape.at(0), 1);
  ASSERT_EQ(shape.at(1), 3);
  ASSERT_EQ(shape.at(2), 5);
  ASSERT_FALSE(shape != shape);
  ASSERT_TRUE((shape != legate::Shape{1, 3, 4}));
}

TEST_P(CreateShapeTest, Ready)
{
  const auto [shape, dim, volume, extents] = GetParam();

  ASSERT_EQ(shape.ndim(), dim);
  ASSERT_EQ(shape.volume(), volume);
  ASSERT_EQ(shape.extents(), extents);
}

TEST_P(EmptyShapeTest, EmptyShape)
{
  const auto& shape = GetParam();

  ASSERT_EQ(shape.ndim(), 0);
  ASSERT_EQ(shape.volume(), 1);
  ASSERT_EQ(shape.extents(), legate::tuple<std::uint64_t>{});
}

TEST_F(ShapeUnit, UnboundShapeToString)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::int32(), LEGATE_MAX_DIM);
  auto shape   = store.shape();

  ASSERT_THAT(shape.to_string(), ::testing::MatchesRegex(R"(Shape\(unbound [0-9]+D\))"));
}

TEST_F(ShapeUnit, ReadyShapeToString)
{
  auto shape = legate::Shape{1, 2, 3};

  ASSERT_THAT(shape.to_string(), ::testing::MatchesRegex(R"(Shape \[[0-9]+(, [0-9]+)*\])"));
}

TEST_F(ShapeUnit, OperatorEqualUnbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto store1  = runtime->create_store(legate::int32(), LEGATE_MAX_DIM);
  auto store2  = runtime->create_store(legate::int32(), /*dim=*/1);
  auto shape1  = store1.shape();
  auto shape2  = store2.shape();

  ASSERT_THAT([&] { static_cast<void>(shape1 == shape2); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
}

TEST_F(ShapeUnit, OperatorEqualBound)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto context        = runtime->find_library(Config::LIBRARY_NAME);
  auto logical_store1 = runtime->create_store(legate::int64(), LEGATE_MAX_DIM);
  auto logical_store2 = runtime->create_store(legate::int64(), LEGATE_MAX_DIM);
  auto logical_store3 = runtime->create_store(legate::uint64(), LEGATE_MAX_DIM);
  auto task           = runtime->create_task(context, UnboundStoreTask::TASK_CONFIG.task_id());

  task.add_output(logical_store1);
  task.add_output(logical_store2);
  task.add_output(logical_store3);
  runtime->submit(std::move(task));

  auto store1 = logical_store1.get_physical_store();
  auto store2 = logical_store2.get_physical_store();
  auto store3 = logical_store3.get_physical_store();

  ASSERT_FALSE(store1.is_unbound_store());
  ASSERT_FALSE(logical_store1.unbound());

  auto bound_shape1 = logical_store1.shape();
  auto bound_shape2 = logical_store2.shape();
  auto bound_shape3 = logical_store3.shape();

  ASSERT_THAT(bound_shape1.to_string(), ::testing::MatchesRegex(R"(Shape\(bound [0-9]+D\))"));
  ASSERT_TRUE(bound_shape1 == bound_shape2);
  ASSERT_FALSE(bound_shape1 == bound_shape3);

  auto other_shape = legate::Shape{1};

  ASSERT_FALSE(bound_shape1 == other_shape);
  ASSERT_FALSE(other_shape == bound_shape2);
}

TEST_F(ShapeUnit, CopyExtentsFromBound)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto runtime_impl = runtime->impl();
  std::array<std::uint64_t, 2> extents{2, 3};
  const auto span        = legate::Span<const std::uint64_t>{extents};
  const auto index_space = runtime_impl->find_or_create_index_space(span);

  legate::detail::Shape source{2};
  source.set_index_space(index_space);
  const auto source_extents = source.extents();

  ASSERT_EQ(source_extents.size(), 2);
  ASSERT_EQ(source_extents[0], 2);
  ASSERT_EQ(source_extents[1], 3);

  legate::detail::Shape target{2};
  target.set_index_space(index_space);
  target.copy_extents_from(source);
  ASSERT_TRUE(target.ready());
  const auto target_extents = target.extents();

  ASSERT_EQ(target_extents.size(), 2);
  ASSERT_EQ(target_extents[0], 2);
  ASSERT_EQ(target_extents[1], 3);
}

}  // namespace shape_test
