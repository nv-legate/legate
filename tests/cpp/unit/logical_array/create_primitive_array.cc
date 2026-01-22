/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/redop/redop.h>

#include <unit/logical_array/utils.h>

namespace logical_array_create_test {

using logical_array_util_test::BOUND_DIM;
using logical_array_util_test::bound_shape_multi_dim;
using logical_array_util_test::UNBOUND_DIM;

namespace {

class TestArrayWithScalarTask : public legate::LegateTask<TestArrayWithScalarTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

class TestArrayWithReductionTask : public legate::LegateTask<TestArrayWithReductionTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);

  static void cpu_variant(legate::TaskContext);
};

/*static*/ void TestArrayWithScalarTask::cpu_variant(legate::TaskContext context)
{
  auto array                        = context.output(0);
  auto nullable                     = context.scalar(0).value<bool>();
  auto store                        = array.data();
  static constexpr std::int32_t DIM = 1;

  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.dim(), DIM);
  ASSERT_EQ(array.type(), legate::int64());
  ASSERT_FALSE(array.nested());

  ASSERT_FALSE(store.is_unbound_store());
  ASSERT_TRUE(store.is_future());
  ASSERT_EQ(store.dim(), DIM);
  ASSERT_EQ(store.type(), legate::int64());

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_FALSE(null_mask.is_unbound_store());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THAT([&] { static_cast<void>(array.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
  }
  ASSERT_THAT([&] { static_cast<void>(array.child(0)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));
  ASSERT_THAT([&] { static_cast<void>(array.as_list_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a list array")));
  ASSERT_THAT([&] { static_cast<void>(array.as_string_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a string array")));
}

/*static*/ void TestArrayWithReductionTask::cpu_variant(legate::TaskContext context)
{
  auto array                        = context.reduction(0);
  auto nullable                     = context.scalar(0).value<bool>();
  auto store                        = array.data();
  static constexpr std::int32_t DIM = 1;

  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.dim(), DIM);
  ASSERT_EQ(array.type(), legate::int64());
  ASSERT_FALSE(array.nested());

  auto op_shape                    = store.shape<DIM>();
  static constexpr auto INIT_VALUE = 10;

  if (!op_shape.empty()) {
    auto reduce_acc = store.reduce_accessor<legate::SumReduction<std::int64_t>, false, DIM>();

    for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
      const legate::Point<DIM> pos{*it};

      reduce_acc.reduce(pos, INIT_VALUE);
    }
  }

  if (!op_shape.empty()) {
    auto read_acc = store.read_accessor<std::int64_t, DIM>();

    for (legate::PointInRectIterator<DIM> it{op_shape}; it.valid(); ++it) {
      ASSERT_EQ(read_acc[*it], INIT_VALUE + 1);
    }
  }
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_create_primitive_array";

  static void registration_callback(legate::Library library)
  {
    TestArrayWithScalarTask::register_variants(library);
    TestArrayWithReductionTask::register_variants(library);
  }
};

class PrimitiveArrayCreateUnit : public RegisterOnceFixture<Config> {};

class PrimitiveNullableTest : public PrimitiveArrayCreateUnit,
                              public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(PrimitiveArrayCreateUnit,
                         PrimitiveNullableTest,
                         ::testing::Values(true, false));

}  // namespace

TEST_P(PrimitiveNullableTest, CreateBoundPrimitiveArray)
{
  const auto nullable = GetParam();
  auto primitive_type = legate::int64();
  auto array =
    logical_array_util_test::create_array_with_type(primitive_type, /*bound=*/true, nullable);

  ASSERT_FALSE(array.unbound());
  ASSERT_EQ(array.dim(), BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_multi_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_multi_dim().volume());
  ASSERT_EQ(array.type(), primitive_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), 0);
  ASSERT_FALSE(array.nested());

  auto store = array.data();

  ASSERT_FALSE(store.unbound());
  ASSERT_EQ(store.dim(), BOUND_DIM);
  ASSERT_EQ(store.extents(), bound_shape_multi_dim().extents());
  ASSERT_EQ(store.volume(), bound_shape_multi_dim().volume());
  ASSERT_EQ(store.type(), primitive_type);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.extents(), array.extents());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THAT([&] { static_cast<void>(array.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
  }
  ASSERT_THAT([&] { static_cast<void>(array.child(0)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));
}

TEST_P(PrimitiveNullableTest, CreateUnboundPrimitiveArray)
{
  const auto nullable = GetParam();
  auto primitive_type = legate::int32();
  auto array =
    logical_array_util_test::create_array_with_type(primitive_type, /*bound=*/false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), UNBOUND_DIM);
  ASSERT_THAT([&] { static_cast<void>(array.extents()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_THAT([&] { static_cast<void>(array.volume()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_EQ(array.type(), primitive_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), 0);
  ASSERT_FALSE(array.nested());

  auto store = array.data();

  ASSERT_TRUE(store.unbound());
  ASSERT_EQ(store.dim(), UNBOUND_DIM);
  ASSERT_THAT([&] { static_cast<void>(store.extents()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_THAT([&] { static_cast<void>(store.volume()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_EQ(store.type(), primitive_type);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THAT([&] { static_cast<void>(null_mask.extents()); },
                ::testing::ThrowsMessage<std::invalid_argument>(
                  ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THAT([&] { static_cast<void>(array.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
  }
  ASSERT_THAT([&] { static_cast<void>(array.child(0)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));
}

TEST_P(PrimitiveNullableTest, CreateBoundLike)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto source =
    logical_array_util_test::create_array_with_type(legate::int64(), /*bound=*/true, nullable);
  auto target1 = runtime->create_array_like(source);

  ASSERT_EQ(source.dim(), target1.dim());
  ASSERT_EQ(source.type(), target1.type());
  ASSERT_EQ(source.extents(), target1.extents());
  ASSERT_EQ(source.volume(), target1.volume());
  ASSERT_EQ(source.unbound(), target1.unbound());
  ASSERT_EQ(source.nullable(), target1.nullable());

  auto target2 = runtime->create_array_like(source, legate::float64());

  ASSERT_EQ(source.dim(), target2.dim());
  ASSERT_EQ(target2.type(), legate::float64());
  ASSERT_EQ(source.extents(), target2.extents());
  ASSERT_EQ(source.volume(), target2.volume());
  ASSERT_EQ(source.unbound(), target2.unbound());
  ASSERT_EQ(source.nullable(), target2.nullable());
}

TEST_P(PrimitiveNullableTest, CreateUnboundLike)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto source =
    logical_array_util_test::create_array_with_type(legate::int64(), /*bound=*/false, nullable);
  auto target1 = runtime->create_array_like(source);

  ASSERT_EQ(source.dim(), target1.dim());
  ASSERT_EQ(source.type(), target1.type());
  ASSERT_THAT([&] { static_cast<void>(target1.extents()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_THAT([&] { static_cast<void>(target1.volume()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_EQ(source.unbound(), target1.unbound());
  ASSERT_EQ(source.nullable(), target1.nullable());

  auto target2 = runtime->create_array_like(source, legate::float64());

  ASSERT_EQ(source.dim(), target2.dim());
  ASSERT_EQ(target2.type(), legate::float64());
  ASSERT_THAT([&] { static_cast<void>(target2.extents()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_THAT([&] { static_cast<void>(target2.volume()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Illegal to access an uninitialized unbound store")));
  ASSERT_EQ(source.unbound(), target2.unbound());
  ASSERT_EQ(source.nullable(), target2.nullable());
}

TEST_P(PrimitiveNullableTest, InvalidCastBoundArray)
{
  auto bound_array =
    logical_array_util_test::create_array_with_type(legate::int64(), /*bound=*/true, GetParam());

  ASSERT_THAT([&] { static_cast<void>(bound_array.as_list_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a list array")));
  ASSERT_THAT([&] { static_cast<void>(bound_array.as_string_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a string array")));
}

TEST_P(PrimitiveNullableTest, InvalidCastUnboundArray)
{
  auto unbound_array =
    logical_array_util_test::create_array_with_type(legate::int64(), /*bound=*/false, GetParam());

  ASSERT_THAT([&] { static_cast<void>(unbound_array.as_list_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a list array")));
  ASSERT_THAT([&] { static_cast<void>(unbound_array.as_string_array()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Array is not a string array")));
}

TEST_F(PrimitiveArrayCreateUnit, CreateNullableArray)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{1}, legate::int64());
  const auto null_mask = runtime->create_store(legate::Shape{1}, legate::bool_());
  const auto array     = runtime->create_nullable_array(store, null_mask);

  ASSERT_EQ(array.dim(), 1);
  ASSERT_EQ(array.type(), legate::int64());
  ASSERT_EQ(array.nullable(), true);
  ASSERT_EQ(array.volume(), 1);
}

TEST_F(PrimitiveArrayCreateUnit, InvalidNullableArrayShape)
{
  auto runtime          = legate::Runtime::get_runtime();
  const auto store      = runtime->create_store(legate::Shape{1}, legate::int64());
  const auto large_mask = runtime->create_store(legate::Shape{4, 3}, legate::bool_());

  // incorrect mask shape
  ASSERT_THAT([&] { return runtime->create_nullable_array(store, large_mask); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Store and null mask must have the same shape")));
}

TEST_F(PrimitiveArrayCreateUnit, InvalidNullableArrayType)
{
  auto runtime          = legate::Runtime::get_runtime();
  const auto store      = runtime->create_store(legate::Shape{1}, legate::int64());
  const auto float_mask = runtime->create_store(legate::Shape{1}, legate::float64());

  // incorrect mask type
  ASSERT_THAT([&] { return runtime->create_nullable_array(store, float_mask); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Null mask must be a boolean type")));
}

TEST_P(PrimitiveNullableTest, PhysicalArray)
{
  const auto nullable = GetParam();
  auto primitive_type = legate::int64();
  auto logical_array =
    logical_array_util_test::create_array_with_type(primitive_type, /*bound=*/true, nullable);
  auto array = logical_array.get_physical_array();

  ASSERT_EQ(array.dim(), BOUND_DIM);
  ASSERT_EQ(array.type(), primitive_type);
  ASSERT_FALSE(array.nested());

  auto shape = legate::Rect<BOUND_DIM>{{0, 0, 0, 0}, {0, 1, 2, 3}};

  ASSERT_EQ(array.shape<BOUND_DIM>(), shape);
  ASSERT_EQ((array.domain().bounds<BOUND_DIM, std::int64_t>()), shape);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.shape<BOUND_DIM>(), array.shape<BOUND_DIM>());
    ASSERT_EQ(null_mask.domain(), array.domain());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THAT([&] { static_cast<void>(array.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
  }
  ASSERT_THAT([&] { static_cast<void>(array.child(0)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));
}

TEST_P(PrimitiveNullableTest, CreateFutureBaseArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto context        = runtime->find_library(Config::LIBRARY_NAME);
  auto task = runtime->create_task(context, TestArrayWithScalarTask::TASK_CONFIG.task_id());
  auto part = task.declare_partition();
  // create null_mask with Future-backed store
  auto logical_array =
    runtime->create_array(legate::Shape{1}, legate::int64(), nullable, /* optimize_scalar */ true);

  task.add_output(logical_array, std::move(part));
  task.add_scalar_arg(legate::Scalar{nullable});
  runtime->submit(std::move(task));
}

TEST_P(PrimitiveNullableTest, CreateFutureBaseArrayReduction)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_array =
    runtime->create_array(legate::Shape{1}, legate::int64(), nullable, /* optimize_scalar */ true);
  auto scalar  = legate::Scalar{std::int64_t{1}};
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, TestArrayWithReductionTask::TASK_CONFIG.task_id());

  runtime->issue_fill(logical_array, scalar);
  task.add_reduction(logical_array, legate::ReductionOpKind::ADD);
  task.add_scalar_arg(legate::Scalar{nullable});
  runtime->submit(std::move(task));
}

}  // namespace logical_array_create_test
