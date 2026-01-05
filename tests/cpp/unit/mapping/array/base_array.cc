/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include <unit/mapping/utils.h>
#include <utilities/utilities.h>

namespace mapping_base_array_test {

using mapping_utils_test::create_test_store;

namespace {

using MappingBaseArrayTest = DefaultFixture;

class BaseArrayNullableTest : public MappingBaseArrayTest,
                              public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(MappingBaseArrayTest, BaseArrayNullableTest, ::testing::Bool());

}  // namespace

TEST_P(BaseArrayNullableTest, Construction)
{
  const auto nullable        = GetParam();
  constexpr std::int32_t DIM = 2;
  auto shape                 = legate::Shape{1, 2};
  auto data_store            = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(shape, legate::int32()));
  std::optional<legate::InternalSharedPtr<legate::mapping::detail::Store>> null_mask;

  if (nullable) {
    null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
      create_test_store(shape, legate::bool_()));
  }

  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, null_mask);

  ASSERT_EQ(base_array->dim(), DIM);
  ASSERT_EQ(base_array->type()->code, legate::int32().code());
  ASSERT_EQ(base_array->nullable(), nullable);
  ASSERT_FALSE(base_array->nested());
  ASSERT_TRUE(base_array->valid());
}

TEST_P(BaseArrayNullableTest, Data)
{
  const auto nullable = GetParam();
  auto data_store     = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{5}, legate::float64()));
  std::optional<legate::InternalSharedPtr<legate::mapping::detail::Store>> null_mask;

  if (nullable) {
    null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
      create_test_store(legate::Shape{5}, legate::bool_()));
  }

  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, null_mask);

  // BaseArray::data() should return the data store
  ASSERT_EQ(base_array->data(), data_store);

  // Also test through base class pointer to ensure virtual dispatch works
  const legate::mapping::detail::Array* array_ptr = base_array.get();
  ASSERT_EQ(array_ptr->data(), data_store);
}

TEST_F(MappingBaseArrayTest, NullMaskNonNullable)
{
  auto data_store = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::int64()));

  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, std::nullopt);

  // Should throw for non-nullable arrays
  ASSERT_THAT([&]() { static_cast<void>(base_array->null_mask()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Invalid to retrieve the null mask of a non-nullable array")));
}

TEST_F(MappingBaseArrayTest, NullMaskNullable)
{
  auto data_store = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::int64()));
  auto null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{8}, legate::bool_()));

  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, null_mask);

  // Should not throw for nullable arrays
  ASSERT_NO_THROW(static_cast<void>(base_array->null_mask()));
  ASSERT_EQ(base_array->null_mask(), null_mask);
}

TEST_F(MappingBaseArrayTest, PopulateStoresNonNullable)
{
  auto data_store = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::int16()));

  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, std::nullopt);

  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Store>> stores;
  base_array->populate_stores(stores);

  ASSERT_THAT(stores, ::testing::ElementsAre(data_store));
}

TEST_F(MappingBaseArrayTest, PopulateStoresNullable)
{
  auto data_store = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::int16()));
  auto null_mask = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{6}, legate::bool_()));

  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, null_mask);

  legate::detail::SmallVector<legate::InternalSharedPtr<legate::mapping::detail::Store>> stores;
  base_array->populate_stores(stores);

  ASSERT_THAT(stores, ::testing::ElementsAre(data_store, null_mask));
}

TEST_F(MappingBaseArrayTest, Unbound)
{
  auto data_store = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{4}, legate::int32()));

  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, std::nullopt);

  ASSERT_FALSE(base_array->unbound());
}

TEST_F(MappingBaseArrayTest, Domain)
{
  auto data_store = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{7, 8}, legate::float32()));
  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, std::nullopt);
  auto domain = base_array->domain();

  ASSERT_EQ(domain.dim, 2);
}

TEST_F(MappingBaseArrayTest, Child)
{
  auto data_store = legate::make_internal_shared<legate::mapping::detail::Store>(
    create_test_store(legate::Shape{3}, legate::int32()));
  auto base_array =
    legate::make_internal_shared<legate::mapping::detail::BaseArray>(data_store, std::nullopt);

  // BaseArray::child() should throw for any index
  ASSERT_THAT([&]() { static_cast<void>(base_array->child(0)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));
  ASSERT_THAT([&]() { static_cast<void>(base_array->child(1)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));

  // Also test through base class pointer
  const legate::mapping::detail::Array* array_ptr = base_array.get();

  ASSERT_THAT([&]() { static_cast<void>(array_ptr->child(0)); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Non-nested array has no child sub-array")));
}

}  // namespace mapping_base_array_test
