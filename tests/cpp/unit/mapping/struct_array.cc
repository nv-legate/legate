/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <unit/mapping/array_test_utils.h>
#include <utilities/utilities.h>

namespace mapping_struct_array_test {

using mapping_array_test::create_test_store;

namespace {

using MappingStructArrayTest = DefaultFixture;

class StructArrayNullableTest : public MappingStructArrayTest,
                                public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(MappingStructArrayTest, StructArrayNullableTest, ::testing::Bool());

}  // namespace

TEST_P(StructArrayNullableTest, Construction)
{
  const auto nullable = GetParam();

  // Create field arrays
  auto field1_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{1, 2}, legate::int32()));
  auto field1 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field1_data, std::nullopt);

  auto field2_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{1, 2}, legate::float64()));
  auto field2 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field2_data, std::nullopt);

  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field1);
  fields.push_back(field2);

  // Create struct type
  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int32(), legate::float64()).impl()};

  // Create null mask if needed
  std::optional<legate::InternalSharedPtr<legate::mapping::detail::Store>> null_mask;

  if (nullable) {
    null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
      create_test_store(legate::Shape{1, 2}, legate::bool_()));
  }

  // Create StructArray
  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, null_mask, std::move(fields));

  ASSERT_EQ(struct_array->dim(), 2);
  ASSERT_EQ(struct_array->kind(), legate::detail::ArrayKind::STRUCT);
  ASSERT_EQ(struct_array->nullable(), nullable);
  ASSERT_TRUE(struct_array->nested());
}

TEST_F(MappingStructArrayTest, Data)
{
  auto field_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{5}, legate::int32()));
  auto field =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int32()).impl()};

  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, std::nullopt, std::move(fields));

  // Nested arrays should not allow data() access - this calls Array::data() base implementation
  // Since StructArray doesn't override data(), it inherits Array::data() which throws
  ASSERT_THAT([&]() { static_cast<void>(struct_array->data()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Data store of a nested array cannot be retrieved")));

  const legate::InternalSharedPtr<legate::mapping::detail::Array> array_ptr = struct_array;

  ASSERT_THAT([&]() { static_cast<void>(array_ptr->data()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Data store of a nested array cannot be retrieved")));
}

TEST_F(MappingStructArrayTest, NullMaskNonNullable)
{
  auto field_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::int64()));
  auto field =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int64()).impl()};

  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, std::nullopt, std::move(fields));

  ASSERT_THAT([&]() { static_cast<void>(struct_array->null_mask()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve the null mask of a non-nullable array")));
}

TEST_F(MappingStructArrayTest, NullMaskNullable)
{
  auto field_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::int64()));
  auto field =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int64()).impl()};

  auto null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::bool_()));

  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, null_mask, std::move(fields));

  ASSERT_NO_THROW(static_cast<void>(struct_array->null_mask()));
  ASSERT_EQ(struct_array->null_mask(), null_mask);
}

TEST_F(MappingStructArrayTest, Child)
{
  auto field1_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::int32()));
  auto field1 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field1_data, std::nullopt);
  auto field2_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::float32()));
  auto field2 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field2_data, std::nullopt);
  auto field3_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::int16()));
  auto field3 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field3_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field1);
  fields.push_back(field2);
  fields.push_back(field3);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int32(), legate::float32(), legate::int16()).impl()};
  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, std::nullopt, std::move(fields));

  // Should be able to access 3 children
  ASSERT_NO_THROW(static_cast<void>(struct_array->child(0)));
  ASSERT_NO_THROW(static_cast<void>(struct_array->child(1)));
  ASSERT_NO_THROW(static_cast<void>(struct_array->child(2)));
  ASSERT_THAT([&]() { static_cast<void>(struct_array->child(3)); },
              ::testing::ThrowsMessage<std::out_of_range>(::testing::_));

  // Verify children using ElementsAre
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> children;
  children.push_back(struct_array->child(0));
  children.push_back(struct_array->child(1));
  children.push_back(struct_array->child(2));

  ASSERT_THAT(children, ::testing::ElementsAre(field1, field2, field3));
}

TEST_F(MappingStructArrayTest, PopulateStores)
{
  auto field1_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{4}, legate::int8()));
  auto field1 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field1_data, std::nullopt);
  auto field2_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{4}, legate::uint64()));
  auto field2 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field2_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field1);
  fields.push_back(field2);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int8(), legate::uint64()).impl()};
  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, std::nullopt, std::move(fields));

  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Store>> stores;
  struct_array->populate_stores(stores);

  // Should have stores from all fields
  ASSERT_THAT(stores, ::testing::ElementsAre(field1_data, field2_data));
}

TEST_F(MappingStructArrayTest, Domain)
{
  auto field_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{2, 5}, legate::int32()));
  auto field =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int32()).impl()};
  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, std::nullopt, std::move(fields));
  auto domain = struct_array->domain();

  ASSERT_EQ(domain.dim, 2);
}

TEST_F(MappingStructArrayTest, Unbound)
{
  auto field1_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{7}, legate::int16()));
  auto field1 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field1_data, std::nullopt);
  auto field2_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{7}, legate::uint32()));
  auto field2 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field2_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field1);
  fields.push_back(field2);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int16(), legate::uint32()).impl()};
  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, std::nullopt, std::move(fields));

  // FutureWrapper-based stores are bound, so all fields are bound
  ASSERT_FALSE(struct_array->unbound());
}

TEST_P(StructArrayNullableTest, Valid)
{
  const auto nullable = GetParam();
  auto field_data     = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{7}, legate::int32()));
  auto field =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int32()).impl()};
  std::optional<legate::InternalSharedPtr<legate::mapping::detail::Store>> null_mask;

  if (nullable) {
    null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
      create_test_store(legate::Shape{7}, legate::bool_()));
  }

  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, null_mask, std::move(fields));

  ASSERT_TRUE(struct_array->valid());
  ASSERT_EQ(struct_array->nullable(), nullable);
}

TEST_F(MappingStructArrayTest, Fields)
{
  auto field1_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{3}, legate::int32()));
  auto field1 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field1_data, std::nullopt);
  auto field2_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{3}, legate::float32()));
  auto field2 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field2_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field1);
  fields.push_back(field2);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int32(), legate::float32()).impl()};
  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, std::nullopt, std::move(fields));
  auto fields_span = struct_array->fields();

  ASSERT_EQ(fields_span.size(), 2);
}

TEST_F(MappingStructArrayTest, PopulateStoresNested)
{
  // Test populate_stores with nested fields (BaseArrays with null masks)
  auto field1_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::int32()));
  auto field1_null = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::bool_()));
  auto field1 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field1_data, field1_null);
  auto field2_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::float32()));
  auto field2 =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(field2_data, std::nullopt);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> fields;

  fields.push_back(field1);
  fields.push_back(field2);

  const legate::InternalSharedPtr<legate::detail::Type> struct_type_ptr{
    legate::struct_type(true, legate::int32(), legate::float32()).impl()};
  auto struct_array = legate::make_internal_shared<legate::mapping::detail::StructArray>(
    struct_type_ptr, std::nullopt, std::move(fields));
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Store>> stores;

  struct_array->populate_stores(stores);

  // Should have: field1_data, field1_null, field2_data = 3 stores
  ASSERT_EQ(stores.size(), 3);
}

}  // namespace mapping_struct_array_test
