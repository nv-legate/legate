/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <unit/mapping/utils.h>
#include <utilities/utilities.h>

namespace mapping_list_array_test {

using mapping_utils_test::create_test_store;

namespace {

using MappingListArrayTest = DefaultFixture;

class ListArrayNullableTest : public MappingListArrayTest,
                              public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(MappingListArrayTest, ListArrayNullableTest, ::testing::Bool());

}  // namespace

TEST_P(ListArrayNullableTest, Construction)
{
  const auto nullable = GetParam();

  // Create descriptor (BaseArray with rect type)
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{5}, legate::rect_type(1)));
  std::optional<legate::InternalSharedPtr<legate::mapping::detail::Store>> null_mask;

  if (nullable) {
    null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
      create_test_store(legate::Shape{5}, legate::bool_()));
  }

  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, null_mask);

  // Create vardata (BaseArray with int64 type)
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::int64()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);

  // Create list type - get internal pointer from public Type
  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::int64()).impl()};

  // Create ListArray
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);

  ASSERT_EQ(list_array->dim(), 1);
  ASSERT_EQ(list_array->nullable(), nullable);
  ASSERT_TRUE(list_array->nested());
}

TEST_F(MappingListArrayTest, Data)
{
  // Create descriptor
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{5}, legate::rect_type(1)));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, std::nullopt);

  // Create vardata
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{5}, legate::int32()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);

  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::int32()).impl()};
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);

  // Nested arrays should not allow data() access - this calls Array::data() base implementation
  // Since ListArray doesn't override data(), it inherits Array::data() which throws
  ASSERT_THAT([&]() { static_cast<void>(list_array->data()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Data store of a nested array cannot be retrieved")));

  // Explicitly test through base class reference to ensure we're calling the base implementation
  const legate::mapping::detail::Array& array_ref = *list_array;

  ASSERT_THAT([&]() { static_cast<void>(array_ref.data()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Data store of a nested array cannot be retrieved")));

  // Also verify via InternalSharedPtr<Array> - this is closer to real usage
  const legate::InternalSharedPtr<legate::mapping::detail::Array> array_ptr = list_array;

  ASSERT_THAT([&]() { static_cast<void>(array_ptr->data()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Data store of a nested array cannot be retrieved")));
}

TEST_F(MappingListArrayTest, NullMaskNonNullable)
{
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::rect_type(1)));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, std::nullopt);
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::float64()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);
  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::float64()).impl()};
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);

  ASSERT_THAT([&]() { static_cast<void>(list_array->null_mask()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve the null mask of a non-nullable array")));
}

TEST_F(MappingListArrayTest, NullMaskNullable)
{
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::rect_type(1)));
  auto null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::bool_()));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, null_mask);
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::float64()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);
  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::float64()).impl()};
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);

  ASSERT_NO_THROW(static_cast<void>(list_array->null_mask()));
  ASSERT_EQ(list_array->null_mask(), null_mask);
}

TEST_F(MappingListArrayTest, Child)
{
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::rect_type(1)));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, std::nullopt);
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::int16()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);
  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::int16()).impl()};
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);

  // ListArray has 2 children: descriptor (0) and vardata (1)
  ASSERT_NO_THROW(static_cast<void>(list_array->child(0)));
  ASSERT_NO_THROW(static_cast<void>(list_array->child(1)));
  ASSERT_THAT([&]() { static_cast<void>(list_array->child(2)); },
              ::testing::ThrowsMessage<std::out_of_range>(::testing::_));
  ASSERT_THAT([&]() { static_cast<void>(list_array->child(3)); },
              ::testing::ThrowsMessage<std::out_of_range>(::testing::_));

  // Verify children are correct using ElementsAre
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Array>> children;
  children.push_back(list_array->child(0));
  children.push_back(list_array->child(1));

  ASSERT_THAT(children, ::testing::ElementsAre(descriptor, vardata));
}

TEST_F(MappingListArrayTest, PopulateStoresNonNullable)
{
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{4}, legate::rect_type(1)));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, std::nullopt);
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{4}, legate::int8()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);
  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::int8()).impl()};
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Store>> stores;

  list_array->populate_stores(stores);

  // Non-nullable: desc_data + var_data = 2 stores
  ASSERT_THAT(stores, ::testing::ElementsAre(desc_data, var_data));
}

TEST_F(MappingListArrayTest, PopulateStoresNullable)
{
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{4}, legate::rect_type(1)));
  auto null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{4}, legate::bool_()));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, null_mask);
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{4}, legate::int8()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);
  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::int8()).impl()};
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);
  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Store>> stores;

  list_array->populate_stores(stores);

  // Nullable: desc_data + null_mask + var_data = 3 stores
  ASSERT_THAT(stores, ::testing::ElementsAre(desc_data, null_mask, var_data));
}

TEST_F(MappingListArrayTest, Domain)
{
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{3}, legate::rect_type(1)));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, std::nullopt);
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{3}, legate::uint32()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);
  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::uint32()).impl()};
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);
  auto domain = list_array->domain();

  ASSERT_EQ(domain.dim, 1);
}

TEST_F(MappingListArrayTest, Unbound)
{
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{7}, legate::rect_type(1)));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, std::nullopt);

  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{7}, legate::float32()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);

  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::float32()).impl()};

  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);

  // FutureWrapper-based stores are bound, so both descriptor and vardata are bound
  // ListArray::unbound() returns descriptor()->unbound() || vardata()->unbound()
  ASSERT_FALSE(list_array->unbound());
}

TEST_F(MappingListArrayTest, Valid)
{
  auto desc_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{7}, legate::rect_type(1)));
  auto descriptor =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(desc_data, std::nullopt);
  auto var_data = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{7}, legate::float32()));
  auto vardata =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(var_data, std::nullopt);
  const legate::InternalSharedPtr<legate::detail::Type> list_type_ptr{
    legate::list_type(legate::float32()).impl()};
  auto list_array = legate::make_internal_shared<legate::mapping::detail::ListArray>(
    list_type_ptr, descriptor, vardata);

  ASSERT_TRUE(list_array->valid());
}

}  // namespace mapping_list_array_test
