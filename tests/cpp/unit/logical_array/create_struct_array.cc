/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/logical_arrays/struct_logical_array.h>

#include <unit/logical_array/utils.h>

namespace logical_array_create_test {

using logical_array_util_test::BOUND_DIM;
using logical_array_util_test::bound_shape_multi_dim;
using logical_array_util_test::UNBOUND_DIM;

namespace {

class StructArrayCreateUnit : public DefaultFixture {};

class StructNullableTest : public StructArrayCreateUnit,
                           public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(StructArrayCreateUnit, StructNullableTest, ::testing::Values(true, false));

// Empty task for testing struct array as output
struct StructOutputTask : public legate::LegateTask<StructOutputTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class StructOutputConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_struct_output";

  static void registration_callback(legate::Library library)
  {
    StructOutputTask::register_variants(library);
  }
};

class StructOutputUnit : public RegisterOnceFixture<StructOutputConfig> {};

}  // namespace

TEST_P(StructNullableTest, Creation)
{
  const auto nullable   = GetParam();
  auto runtime          = legate::Runtime::get_runtime();
  const auto int_type   = legate::int64();
  const auto float_type = legate::float32();
  const auto& shape     = bound_shape_multi_dim();

  auto int_array   = runtime->create_array(shape, int_type);
  auto float_array = runtime->create_array(shape, float_type);
  const std::optional<legate::LogicalStore> null_mask =
    nullable ? std::make_optional(runtime->create_store(shape, legate::bool_())) : std::nullopt;
  auto array = runtime->create_struct_array({int_array, float_array}, null_mask);

  ASSERT_EQ(array.dim(), BOUND_DIM);
  ASSERT_EQ(array.type().code(), legate::Type::Code::STRUCT);
  ASSERT_EQ(array.extents(), shape.extents());
  ASSERT_EQ(array.volume(), shape.volume());
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), 2);
  ASSERT_TRUE(array.nested());
}

TEST_P(StructNullableTest, Bound)
{
  const auto nullable = GetParam();
  const auto& st_type = logical_array_util_test::struct_type();
  auto num_fields     = st_type.num_fields();
  auto array = logical_array_util_test::create_array_with_type(st_type, /*bound=*/true, nullable);

  ASSERT_FALSE(array.unbound());
  ASSERT_EQ(array.dim(), BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_multi_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_multi_dim().volume());
  ASSERT_EQ(array.type(), st_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), num_fields);
  ASSERT_TRUE(array.nested());

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
  ASSERT_THAT(
    [&] { static_cast<void>(array.child(num_fields)); },
    ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr("inplace_vector::at")));
  for (std::uint32_t idx = 0; idx < num_fields; ++idx) {
    auto field_type     = st_type.field_type(idx);
    auto field_subarray = array.child(idx);

    ASSERT_EQ(field_subarray.unbound(), array.unbound());
    ASSERT_EQ(field_subarray.dim(), array.dim());
    ASSERT_EQ(field_subarray.extents(), array.extents());
    ASSERT_EQ(field_subarray.volume(), array.volume());
    ASSERT_EQ(field_subarray.type(), field_type);
    // There'd be only one null mask for the whole struct array
    ASSERT_EQ(field_subarray.nullable(), false);
    ASSERT_THAT([&] { static_cast<void>(field_subarray.null_mask()); },
                ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                  "Invalid to retrieve the null mask of a non-nullable array")));
    ASSERT_EQ(field_subarray.num_children(), 0);
  }
}

TEST_P(StructNullableTest, Unbound)
{
  const auto nullable = GetParam();
  const auto& st_type = logical_array_util_test::struct_type();
  auto num_fields     = st_type.num_fields();
  auto array = logical_array_util_test::create_array_with_type(st_type, /*bound=*/false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), UNBOUND_DIM);
  ASSERT_EQ(array.type(), st_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), num_fields);

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
                ::testing::HasSubstr("Invalid to retrieve a sub-array of an unbound array")));
}

TEST_F(StructArrayCreateUnit, InvalidCreation)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto arr_rect1 = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int64 = runtime->create_array(legate::Shape{2}, legate::int64());
  const auto null_mask = runtime->create_store(legate::Shape{1}, legate::bool_());

  // fields of inconsistent shape
  ASSERT_THAT([&] { return runtime->create_struct_array({arr_rect1, arr_int64}, null_mask); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("All fields must have the same shape")));
}

TEST_F(StructArrayCreateUnit, InvalidType)
{
  // Create a non-STRUCT type (INT32)
  auto invalid_type = legate::int32().impl();

  // null_mask and fields can be empty for this test since we expect
  // the type check to fail before they are used
  ASSERT_THAT(
    [&] { static_cast<void>(legate::detail::StructLogicalArray{invalid_type, std::nullopt, {}}); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("expected struct type")));
}

TEST_F(StructArrayCreateUnit, IsMapped)
{
  auto runtime          = legate::Runtime::get_runtime();
  const auto int_type   = legate::int64();
  const auto float_type = legate::float32();
  const auto& shape     = bound_shape_multi_dim();

  auto int_array   = runtime->create_array(shape, int_type);
  auto float_array = runtime->create_array(shape, float_type);
  auto array       = runtime->create_struct_array({int_array, float_array}, std::nullopt);

  // Call is_mapped() on the internal StructLogicalArray
  ASSERT_FALSE(array.impl()->is_mapped());
}

TEST_F(StructArrayCreateUnit, Broadcast)
{
  auto runtime          = legate::Runtime::get_runtime();
  const auto int_type   = legate::int64();
  const auto float_type = legate::float32();
  // Shape with dim 0 having size 1 so we can broadcast it
  const auto shape = legate::Shape{1, 2, 3};

  auto int_array   = runtime->create_array(shape, int_type);
  auto float_array = runtime->create_array(shape, float_type);
  auto array       = runtime->create_struct_array({int_array, float_array}, std::nullopt);

  // Broadcast dimension 0 from size 1 to size 9
  auto broadcasted = array.broadcast(/*dim=*/0, /*dim_size=*/9);

  ASSERT_EQ(broadcasted.dim(), 3);
  ASSERT_EQ(broadcasted.extents()[0], 9);
  ASSERT_EQ(broadcasted.extents()[1], 2);
  ASSERT_EQ(broadcasted.extents()[2], 3);
  ASSERT_EQ(broadcasted.type().code(), legate::Type::Code::STRUCT);
  ASSERT_EQ(broadcasted.num_children(), 2);

  // Verify children types are preserved
  ASSERT_EQ(broadcasted.child(0).type(), int_type);
  ASSERT_EQ(broadcasted.child(1).type(), float_type);

  // Verify children dimensions are also broadcasted
  ASSERT_EQ(broadcasted.child(0).extents()[0], 9);
  ASSERT_EQ(broadcasted.child(1).extents()[0], 9);
}

// Test non-nullable struct array as output with multiple fields
TEST_F(StructOutputUnit, NonNullableOutput)
{
  auto runtime          = legate::Runtime::get_runtime();
  const auto library    = runtime->find_library(StructOutputConfig::LIBRARY_NAME);
  const auto int_type   = legate::int64();
  const auto float_type = legate::float32();
  const auto shape      = legate::Shape{10};

  auto int_array   = runtime->create_array(shape, int_type);
  auto float_array = runtime->create_array(shape, float_type);
  // Create struct array with multiple fields (non-nullable)
  auto struct_arr = runtime->create_struct_array({int_array, float_array}, std::nullopt);

  auto task = runtime->create_task(library, StructOutputTask::TASK_CONFIG.task_id());
  task.add_output(struct_arr);
  runtime->submit(std::move(task));
}

// Test nullable struct array with scalar storage null_mask as output
TEST_F(StructOutputUnit, ScalarStorageNullMaskOutput)
{
  auto runtime        = legate::Runtime::get_runtime();
  const auto library  = runtime->find_library(StructOutputConfig::LIBRARY_NAME);
  const auto int_type = legate::int64();
  const auto shape    = legate::Shape{1};

  // Create a scalar storage null_mask by using optimize_scalar=true
  auto null_mask = runtime->create_store(shape, legate::bool_(), /*optimize_scalar=*/true);
  ASSERT_TRUE(null_mask.has_scalar_storage());

  auto int_array = runtime->create_array(shape, int_type);
  // Create struct array with scalar storage null_mask
  auto struct_arr = runtime->create_struct_array({int_array}, null_mask);

  auto task = runtime->create_task(library, StructOutputTask::TASK_CONFIG.task_id());
  task.add_output(struct_arr);
  runtime->submit(std::move(task));
}

}  // namespace logical_array_create_test
