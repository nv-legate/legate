/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/redop/redop.h>

#include <gtest/gtest.h>

#include <unit/logical_array/utils.h>
#include <utilities/utilities.h>

namespace logical_array_create_test {

namespace {

constexpr auto BOUND_DIM                = 4;
constexpr auto VARILABLE_TYPE_BOUND_DIM = 1;
constexpr auto UNBOUND_DIM              = 1;
constexpr auto NUM_CHILDREN             = 2;

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
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.as_list_array()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.as_string_array()), std::invalid_argument);
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
  static constexpr std::string_view LIBRARY_NAME = "test_create_logical_array";

  static void registration_callback(legate::Library library)
  {
    TestArrayWithScalarTask::register_variants(library);
    TestArrayWithReductionTask::register_variants(library);
  }
};

class LogicalArrayCreateUnit : public RegisterOnceFixture<Config> {};

const legate::Shape& bound_shape_multi_dim()
{
  static const auto shape = legate::Shape{1, 2, 3, 4};

  return shape;
}

const legate::Shape& bound_shape_single_dim()
{
  static const auto shape = legate::Shape{10};

  return shape;
}

class NullableTest : public LogicalArrayCreateUnit, public ::testing::WithParamInterface<bool> {};

class CreateInvalidArrayTest
  : public LogicalArrayCreateUnit,
    public ::testing::WithParamInterface<std::tuple<const legate::LogicalArray /* descriptor */,
                                                    const legate::LogicalArray /* vardata */>> {};

INSTANTIATE_TEST_SUITE_P(LogicalArrayCreateUnit, NullableTest, ::testing::Values(true, false));

std::vector<std::tuple<const legate::LogicalArray, const legate::LogicalArray>>
generate_invalid_arrays()
{
  legate::start();
  auto runtime                 = legate::Runtime::get_runtime();
  const auto arr_unbound_rect1 = runtime->create_array(legate::rect_type(UNBOUND_DIM));
  const auto arr_unbound_int8  = runtime->create_array(legate::int8());
  const auto arr_rect1         = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int8          = runtime->create_array(legate::Shape{1}, legate::int8());
  const auto arr_int64         = runtime->create_array(legate::Shape{1}, legate::int64());
  const auto arr_4d_rect1 =
    runtime->create_array(bound_shape_multi_dim(), legate::rect_type(BOUND_DIM));
  const auto arr_4d_int8 = runtime->create_array(bound_shape_multi_dim(), legate::int8());
  const auto arr_nullable_int8 =
    runtime->create_array(legate::Shape{1}, legate::int8(), /*nullable=*/true);

  return {
    std::make_tuple(arr_unbound_rect1, arr_int8) /* descriptor: unbound sub-array */,
    std::make_tuple(arr_rect1, arr_unbound_int8) /* vardata: unbound sub-array */,
    std::make_tuple(arr_4d_rect1, arr_int8) /* descriptor: multi-dimensional sub-array */,
    std::make_tuple(arr_rect1, arr_4d_int8) /* vardata: multi-dimensional sub-array */,
    std::make_tuple(arr_int64, arr_int8) /* incorrect descriptor type */,
    std::make_tuple(arr_rect1, arr_nullable_int8) /* nullable vardata */
  };
}

INSTANTIATE_TEST_SUITE_P(LogicalArrayCreateUnit,
                         CreateInvalidArrayTest,
                         ::testing::ValuesIn(generate_invalid_arrays()));

}  // namespace

TEST_P(NullableTest, CreateBoundPrimitiveArray)
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
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

TEST_P(NullableTest, CreateUnboundPrimitiveArray)
{
  const auto nullable = GetParam();
  auto primitive_type = legate::int32();
  auto array =
    logical_array_util_test::create_array_with_type(primitive_type, /*bound=*/false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), UNBOUND_DIM);
  ASSERT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
  ASSERT_EQ(array.type(), primitive_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), 0);
  ASSERT_FALSE(array.nested());

  auto store = array.data();

  ASSERT_TRUE(store.unbound());
  ASSERT_EQ(store.dim(), UNBOUND_DIM);
  ASSERT_THROW(static_cast<void>(store.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(store.volume()), std::invalid_argument);
  ASSERT_EQ(store.type(), primitive_type);

  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

TEST_P(NullableTest, CreateBoundListArray)
{
  const auto nullable = GetParam();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto array = logical_array_util_test::create_array_with_type(arr_type, /*bound=*/true, nullable);

  // List arrays are unbound even with the fixed extents
  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_single_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_single_dim().volume());
  ASSERT_EQ(array.type(), arr_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.extents(), array.extents());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto list_array = array.as_list_array();

  // Sub-arrays of list arrays can be retrieved only when they are initialized first
  ASSERT_THROW(static_cast<void>(list_array.descriptor()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(list_array.vardata()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(list_array.child(NUM_CHILDREN)), std::invalid_argument);
}

TEST_P(NullableTest, CreateUnboundListArray)
{
  const auto nullable = GetParam();
  auto arr_type       = legate::list_type(legate::int64()).as_list_type();
  auto array = logical_array_util_test::create_array_with_type(arr_type, /*bound=*/false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
  ASSERT_EQ(array.type(), arr_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto list_array = array.as_list_array();

  // Sub-arrays of list arrays can be retrieved only when they are initialized first
  ASSERT_THROW(static_cast<void>(list_array.descriptor()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(list_array.vardata()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(list_array.child(NUM_CHILDREN)), std::invalid_argument);
}

TEST_P(NullableTest, CreateListArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto type           = legate::list_type(legate::int64());
  auto descriptor     = runtime->create_array(
    bound_shape_single_dim(), legate::rect_type(VARILABLE_TYPE_BOUND_DIM), nullable);
  auto vardata = runtime->create_array(bound_shape_single_dim(), legate::int64());
  auto array   = runtime->create_list_array(descriptor, vardata, type);

  ASSERT_FALSE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_single_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_single_dim().volume());
  // compare code here since UID of ListType differ between different objects
  ASSERT_EQ(array.type().code(), legate::Type::Code::LIST);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);

  auto list_array = array.as_list_array();
  // Sub-arrays can be accessed
  ASSERT_NO_THROW(static_cast<void>(list_array.descriptor()));
  ASSERT_NO_THROW(static_cast<void>(list_array.vardata()));
  for (std::uint32_t idx = 0; idx < NUM_CHILDREN; ++idx) {
    ASSERT_NO_THROW(static_cast<void>(list_array.child(idx)));
  }
  ASSERT_THROW(static_cast<void>(list_array.child(NUM_CHILDREN)), std::out_of_range);
}

TEST_P(NullableTest, CreateStructArray)
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

TEST_P(NullableTest, CreateBoundStringArray)
{
  const auto nullable = GetParam();
  auto str_type       = legate::string_type();
  auto array = logical_array_util_test::create_array_with_type(str_type, /*bound=*/true, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), VARILABLE_TYPE_BOUND_DIM);
  ASSERT_EQ(array.extents(), bound_shape_single_dim().extents());
  ASSERT_EQ(array.volume(), bound_shape_single_dim().volume());
  ASSERT_EQ(array.type(), str_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_EQ(null_mask.extents(), array.extents());
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto string_array = array.as_string_array();

  // Sub-arrays of string arrays can be retrieved only when they are initialized first
  ASSERT_THROW(static_cast<void>(string_array.chars()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(string_array.offsets()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(string_array.child(NUM_CHILDREN)), std::invalid_argument);
}

TEST_P(NullableTest, CreateUnboundStringArray)
{
  const auto nullable = GetParam();
  auto str_type       = legate::string_type();
  auto array = logical_array_util_test::create_array_with_type(str_type, /*bound=*/false, nullable);

  ASSERT_TRUE(array.unbound());
  ASSERT_EQ(array.dim(), UNBOUND_DIM);
  ASSERT_THROW(static_cast<void>(array.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(array.volume()), std::invalid_argument);
  ASSERT_EQ(array.type(), str_type);
  ASSERT_EQ(array.nullable(), nullable);
  ASSERT_EQ(array.num_children(), NUM_CHILDREN);
  ASSERT_TRUE(array.nested());

  ASSERT_THROW(static_cast<void>(array.data()), std::invalid_argument);
  if (nullable) {
    auto null_mask = array.null_mask();

    ASSERT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }

  auto string_array = array.as_string_array();

  // Sub-arrays of string arrays can be retrieved only when they are initialized first
  ASSERT_THROW(static_cast<void>(string_array.chars()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(string_array.offsets()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(string_array.child(NUM_CHILDREN)), std::invalid_argument);
}

TEST_P(NullableTest, CreateBoundStructArray)
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
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(num_fields)), std::out_of_range);
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
    ASSERT_THROW(static_cast<void>(field_subarray.null_mask()), std::invalid_argument);
    ASSERT_EQ(field_subarray.num_children(), 0);
  }
}

TEST_P(NullableTest, CreateUnboundStructArray)
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

    ASSERT_THROW(static_cast<void>(null_mask.extents()), std::invalid_argument);
    ASSERT_EQ(null_mask.type(), legate::bool_());
    ASSERT_EQ(null_mask.dim(), array.dim());
  } else {
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

TEST_P(NullableTest, CreateBoundLike)
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

TEST_P(NullableTest, CreateUnboundLike)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto source =
    logical_array_util_test::create_array_with_type(legate::int64(), /*bound=*/false, nullable);
  auto target1 = runtime->create_array_like(source);

  ASSERT_EQ(source.dim(), target1.dim());
  ASSERT_EQ(source.type(), target1.type());
  ASSERT_THROW(static_cast<void>(target1.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(target1.volume()), std::invalid_argument);
  ASSERT_EQ(source.unbound(), target1.unbound());
  ASSERT_EQ(source.nullable(), target1.nullable());

  auto target2 = runtime->create_array_like(source, legate::float64());

  ASSERT_EQ(source.dim(), target2.dim());
  ASSERT_EQ(target2.type(), legate::float64());
  ASSERT_THROW(static_cast<void>(target2.extents()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(target2.volume()), std::invalid_argument);
  ASSERT_EQ(source.unbound(), target2.unbound());
  ASSERT_EQ(source.nullable(), target2.nullable());
}

TEST_P(NullableTest, InvalidCastBoundArray)
{
  auto bound_array =
    logical_array_util_test::create_array_with_type(legate::int64(), /*bound=*/true, GetParam());

  ASSERT_THROW(static_cast<void>(bound_array.as_list_array()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(bound_array.as_string_array()), std::invalid_argument);
}

TEST_P(NullableTest, InvalidCastUnboundArray)
{
  auto unbound_array =
    logical_array_util_test::create_array_with_type(legate::int64(), /*bound=*/false, GetParam());

  ASSERT_THROW(static_cast<void>(unbound_array.as_list_array()), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(unbound_array.as_string_array()), std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, InvalidCreateListArray)
{
  auto runtime = legate::Runtime::get_runtime();

  // Multi-dimensional list arrays are not allowed
  ASSERT_THROW(static_cast<void>(runtime->create_array(bound_shape_multi_dim(),
                                                       legate::list_type(legate::int64()))),
               std::invalid_argument);
  ASSERT_THROW(
    static_cast<void>(runtime->create_array(legate::list_type(legate::int64()), BOUND_DIM)),
    std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, InvalidCreateStringArray)
{
  auto runtime = legate::Runtime::get_runtime();

  // Multi-dimensional string arrays are not allowed
  ASSERT_THROW(
    static_cast<void>(runtime->create_array(bound_shape_multi_dim(), legate::string_type())),
    std::invalid_argument);
  ASSERT_THROW(static_cast<void>(runtime->create_array(legate::string_type(), BOUND_DIM)),
               std::invalid_argument);
}

TEST_P(CreateInvalidArrayTest, StringArray)
{
  const auto [descriptor, vardata] = GetParam();
  auto runtime                     = legate::Runtime::get_runtime();

  ASSERT_THROW(static_cast<void>(runtime->create_string_array(descriptor, vardata)),
               std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, InvalidStringArray)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto arr_rect1 = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int64 = runtime->create_array(legate::Shape{1}, legate::int64());

  // incorrect vardata type
  ASSERT_THROW(static_cast<void>(runtime->create_string_array(arr_rect1, arr_int64)),
               std::invalid_argument);
}

TEST_P(CreateInvalidArrayTest, ListArray)
{
  const auto [descriptor, vardata] = GetParam();
  auto runtime                     = legate::Runtime::get_runtime();

  ASSERT_THROW(static_cast<void>(runtime->create_list_array(descriptor, vardata)),
               std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, InvalidListArray)
{
  auto runtime         = legate::Runtime::get_runtime();
  const auto arr_rect1 = runtime->create_array(legate::Shape{1}, legate::rect_type(1));
  const auto arr_int64 = runtime->create_array(legate::Shape{1}, legate::int64());

  // incorrect type
  ASSERT_THROW(static_cast<void>(runtime->create_list_array(arr_rect1, arr_int64, legate::int64())),
               std::invalid_argument);

  // incorrect vardata type
  ASSERT_THROW(static_cast<void>(runtime->create_list_array(
                 arr_rect1, arr_int64, legate::list_type(legate::int8()))),
               std::invalid_argument);
}

TEST_F(LogicalArrayCreateUnit, CreateNullableArray)
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

TEST_F(LogicalArrayCreateUnit, InvalidNullableArrayShape)
{
  auto runtime          = legate::Runtime::get_runtime();
  const auto store      = runtime->create_store(legate::Shape{1}, legate::int64());
  const auto large_mask = runtime->create_store(legate::Shape{4, 3}, legate::bool_());

  // incorrect mask shape
  ASSERT_THAT([&] { return runtime->create_nullable_array(store, large_mask); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Store and null mask must have the same shape")));
}

TEST_F(LogicalArrayCreateUnit, InvalidNullableArrayType)
{
  auto runtime          = legate::Runtime::get_runtime();
  const auto store      = runtime->create_store(legate::Shape{1}, legate::int64());
  const auto float_mask = runtime->create_store(legate::Shape{1}, legate::float64());

  // incorrect mask type
  ASSERT_THAT([&] { return runtime->create_nullable_array(store, float_mask); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Null mask must be a boolean type")));
}

TEST_F(LogicalArrayCreateUnit, InvalidStructArray)
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

TEST_P(NullableTest, PhsicalArray)
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
    ASSERT_THROW(static_cast<void>(array.null_mask()), std::invalid_argument);
  }
  ASSERT_THROW(static_cast<void>(array.child(0)), std::invalid_argument);
}

TEST_P(NullableTest, ListPhysicalArray)
{
  const auto nullable = GetParam();
  auto runtime        = legate::Runtime::get_runtime();
  auto type           = legate::list_type(legate::int64());
  auto descriptor     = runtime->create_array(
    bound_shape_single_dim(), legate::rect_type(VARILABLE_TYPE_BOUND_DIM), nullable);
  auto vardata = runtime->create_array(bound_shape_single_dim(), legate::int64());
  auto array   = runtime->create_list_array(descriptor, vardata, type);

  auto list_array          = array.as_list_array();
  auto physical_array      = list_array.get_physical_array();
  auto list_physical_array = physical_array.as_list_array();
  ASSERT_NO_THROW(static_cast<void>(list_physical_array.descriptor()));
  ASSERT_NO_THROW(static_cast<void>(list_physical_array.vardata()));
  ASSERT_EQ(list_physical_array.nullable(), nullable);
  ASSERT_EQ(list_physical_array.type(), type);
  ASSERT_EQ(list_physical_array.dim(), VARILABLE_TYPE_BOUND_DIM);
}

TEST_P(NullableTest, CreateFutureBaseArray)
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

TEST_P(NullableTest, CreateFutureBaseArrayReduction)
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
