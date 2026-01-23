/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace physical_array_create_test {

namespace {

// Helper to extract DIM from Rect<DIM, CoordT>
template <typename T>
struct RectDim;

template <std::int32_t D, typename CoordT>
struct RectDim<Realm::Rect<D, CoordT>> {
  static constexpr std::int32_t VALUE = D;
};

class UnboundArrayTask : public legate::LegateTask<UnboundArrayTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_create_physical_array";

  static void registration_callback(legate::Library library)
  {
    UnboundArrayTask::register_variants(library);
  }
};

class CreatePhysicalArrayUnit : public RegisterOnceFixture<Config> {};

class NullableCreateArrayTest : public CreatePhysicalArrayUnit,
                                public ::testing::WithParamInterface<bool> {};

using BoundRectVariant = std::variant<legate::Rect<2>, legate::Rect<3>>;

class BoundPhysicalArrayTest
  : public CreatePhysicalArrayUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape, bool, BoundRectVariant>> {};

INSTANTIATE_TEST_SUITE_P(CreatePhysicalArrayUnit,
                         NullableCreateArrayTest,
                         ::testing::Values(true, false));

INSTANTIATE_TEST_SUITE_P(
  CreatePhysicalArrayUnit,
  BoundPhysicalArrayTest,
  ::testing::Values(
    std::make_tuple(legate::Shape{2, 4}, true, BoundRectVariant{legate::Rect<2>{{0, 0}, {1, 3}}}),
    std::make_tuple(legate::Shape{2, 4}, false, BoundRectVariant{legate::Rect<2>{{0, 0}, {1, 3}}}),
    std::make_tuple(legate::Shape{2, 3, 4},
                    true,
                    BoundRectVariant{legate::Rect<3>{{0, 0, 0}, {1, 2, 3}}}),
    std::make_tuple(legate::Shape{2, 3, 4},
                    false,
                    BoundRectVariant{legate::Rect<3>{{0, 0, 0}, {1, 2, 3}}})));

/*static*/ void UnboundArrayTask::cpu_variant(legate::TaskContext context)
{
  auto array                        = context.output(0);
  auto nullable                     = context.scalar(0).value<bool>();
  auto store                        = array.data();
  static constexpr std::int32_t DIM = 3;

  ASSERT_TRUE(store.is_unbound_store());
  ASSERT_NO_THROW(static_cast<void>(
    store.create_output_buffer<std::uint32_t, DIM>(legate::Point<DIM>{10}, true)));

  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.dim(), DIM);
  ASSERT_EQ(array.type(), legate::uint32());
  ASSERT_FALSE(array.nested());
  ASSERT_THAT([&]() { static_cast<void>(array.shape<DIM>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve the domain of an unbound store")));
  ASSERT_THAT([&]() { static_cast<void>(array.domain()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve the domain of an unbound store")));

  ASSERT_TRUE(store.is_unbound_store());
  ASSERT_FALSE(store.is_future());
  ASSERT_EQ(store.dim(), DIM);
  ASSERT_THAT([&]() { static_cast<void>(store.shape<DIM>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve the domain of an unbound store")));
  ASSERT_THAT([&]() { static_cast<void>(store.domain()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve the domain of an unbound store")));
  ASSERT_EQ(store.type(), legate::uint32());

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_TRUE(null_mask.is_unbound_store());
    ASSERT_NO_THROW(
      static_cast<void>(null_mask.create_output_buffer<bool, DIM>(legate::Point<DIM>{10}, true)));
    ASSERT_THAT([&]() { static_cast<void>(null_mask.shape<DIM>()); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                  ::testing::HasSubstr("Invalid to retrieve the domain of an unbound store")));
    ASSERT_THAT([&]() { static_cast<void>(null_mask.domain()); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                  ::testing::HasSubstr("Invalid to retrieve the domain of an unbound store")));
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THAT([&]() { static_cast<void>(array.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
  }
  ASSERT_THAT([&]() { static_cast<void>(array.child(0)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));
  ASSERT_THAT([&]() { static_cast<void>(array.as_list_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a list array")));
  ASSERT_THAT([&]() { static_cast<void>(array.as_string_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a string array")));
}

}  // namespace

TEST_P(BoundPhysicalArrayTest, Create)
{
  const auto [shape, nullable, bound_rect_variant] = GetParam();
  auto runtime                                     = legate::Runtime::get_runtime();
  auto type                                        = legate::int64();
  auto logical_array                               = runtime->create_array(shape, type, nullable);
  auto array                                       = logical_array.get_physical_array();

  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.type(), type);
  ASSERT_FALSE(array.nested());

  auto store = array.data();

  ASSERT_FALSE(store.is_unbound_store());
  ASSERT_FALSE(store.is_future());
  ASSERT_EQ(store.type(), type);

  const auto is_nullable = nullable;
  std::visit(
    [&, is_nullable](const auto& bound_rect) {
      constexpr std::int32_t DIM = RectDim<std::decay_t<decltype(bound_rect)>>::VALUE;
      ASSERT_EQ(array.dim(), DIM);
      ASSERT_EQ(array.shape<DIM>(), bound_rect);
      ASSERT_EQ((array.domain().bounds<DIM, std::int64_t>()), bound_rect);
      ASSERT_EQ(store.dim(), DIM);
      ASSERT_EQ(store.shape<DIM>(), bound_rect);

      if (is_nullable) {
        auto null_mask = array.null_mask();

        ASSERT_EQ(null_mask.shape<DIM>(), array.shape<DIM>());
        ASSERT_EQ(null_mask.domain(), array.domain());
        ASSERT_EQ(null_mask.type(), legate::bool_());
        ASSERT_EQ(null_mask.dim(), array.dim());
      }
    },
    bound_rect_variant);

  if (!nullable) {
    ASSERT_THAT([&]() { static_cast<void>(array.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
  }
}

TEST_P(NullableCreateArrayTest, InvalidBoundArrayChild)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_array = runtime->create_array(legate::Shape{1, 3, 4}, legate::uint32(), GetParam());
  auto array         = logical_array.get_physical_array();

  ASSERT_THAT([&]() { static_cast<void>(array.child(0)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));
}

TEST_P(NullableCreateArrayTest, InvalidCastBoundArray)
{
  auto runtime = legate::Runtime::get_runtime();
  auto logical_array =
    runtime->create_array(legate::Shape{1, 3, 4, 2}, legate::int32(), GetParam());
  auto array = logical_array.get_physical_array();

  ASSERT_THAT([&]() { static_cast<void>(array.as_list_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a list array")));
  ASSERT_THAT([&]() { static_cast<void>(array.as_string_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a string array")));
}

TEST_P(NullableCreateArrayTest, UnboundArray)
{
  const auto nullable               = GetParam();
  auto runtime                      = legate::Runtime::get_runtime();
  auto context                      = runtime->find_library(Config::LIBRARY_NAME);
  static constexpr std::int32_t DIM = 3;
  auto logical_array                = runtime->create_array(legate::uint32(), DIM, nullable);
  auto task = runtime->create_task(context, UnboundArrayTask::TASK_CONFIG.task_id());
  auto part = task.declare_partition();

  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  runtime->submit(std::move(task));
  ASSERT_FALSE(logical_array.unbound());
}

}  // namespace physical_array_create_test
