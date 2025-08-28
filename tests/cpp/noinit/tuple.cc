/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace tuple_test {

// NOLINTBEGIN(readability-magic-numbers)

template <typename T>
using TupleUnit = ::testing::Test;

using TupleTypeList =
  ::testing::Types<std::vector<std::int32_t>, std::vector<std::uint64_t>, std::vector<float>>;

TYPED_TEST_SUITE(TupleUnit, TupleTypeList, );

TYPED_TEST(TupleUnit, Create)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  ASSERT_FALSE(tuple1.empty());
  ASSERT_EQ(tuple1.size(), 5);

  const legate::tuple<typename TypeParam::value_type> tuple2{};
  ASSERT_TRUE(tuple2.empty());
  ASSERT_EQ(tuple2.size(), 0);
}

TYPED_TEST(TupleUnit, Reserve)
{
  legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  tuple1.reserve(10);
  ASSERT_EQ(tuple1.size(), 5);

  legate::tuple<typename TypeParam::value_type> tuple2{};
  tuple2.reserve(10);
  ASSERT_EQ(tuple2.size(), 0);
}

TYPED_TEST(TupleUnit, ToString)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  ASSERT_EQ(tuple1.to_string(), "(1,2,3,4,5)");

  const legate::tuple<typename TypeParam::value_type> tuple2{};
  ASSERT_EQ(tuple2.to_string(), "()");
}

TYPED_TEST(TupleUnit, Equality)
{
  const std::vector<typename TypeParam::value_type> init_val1{1, 2, 3, 4, 5};
  const std::vector<typename TypeParam::value_type> init_val2{1, 2, 3, 4, 5};
  const std::vector<typename TypeParam::value_type> init_val3{2, 3, 4, 5, 6};
  const std::vector<typename TypeParam::value_type> init_val4{};
  const legate::tuple<typename TypeParam::value_type> tuple1{init_val1};
  const legate::tuple<typename TypeParam::value_type> tuple2{init_val2};
  const legate::tuple<typename TypeParam::value_type> tuple3{init_val3};
  const legate::tuple<typename TypeParam::value_type> tuple4{init_val4};

  auto test_equality = [](const auto& tuple_ipt1,
                          const auto& tuple_ipt2,
                          const auto& init_ipt1,
                          const auto& init_ipt2) {
    ASSERT_EQ(tuple_ipt1 == tuple_ipt2, init_ipt1 == init_ipt2);
    ASSERT_EQ(tuple_ipt1 != tuple_ipt2, init_ipt1 != init_ipt2);
  };
  test_equality(tuple1, tuple2, init_val1, init_val2);
  test_equality(tuple1, tuple3, init_val1, init_val3);
  test_equality(tuple1, tuple4, init_val1, init_val4);
}

TYPED_TEST(TupleUnit, Less)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple3{2, 3, 4, 5, 6};

  // tuple1 vs tuple1
  ASSERT_FALSE(tuple1.less(tuple1));

  // tuple1 vs tuple2
  ASSERT_FALSE(tuple1.less(tuple2));
  ASSERT_FALSE(tuple2.less(tuple1));

  // tuple1 vs tuple3
  ASSERT_TRUE(tuple1.less(tuple3));
  ASSERT_FALSE(tuple3.less(tuple1));
}

TYPED_TEST(TupleUnit, LessEqual)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple3{2, 3, 4, 5, 6};

  // tuple1 vs tuple1
  ASSERT_TRUE(tuple1.less_equal(tuple1));

  // tuple1 vs tuple2
  ASSERT_TRUE(tuple1.less_equal(tuple2));
  ASSERT_TRUE(tuple2.less_equal(tuple1));

  // tuple1 vs tuple3
  ASSERT_TRUE(tuple1.less_equal(tuple3));
  ASSERT_FALSE(tuple3.less_equal(tuple1));
}

TYPED_TEST(TupleUnit, Greater)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple3{2, 3, 4, 5, 6};

  // tuple1 vs tuple1
  ASSERT_FALSE(tuple1.greater(tuple1));

  // tuple1 vs tuple2
  ASSERT_FALSE(tuple1.greater(tuple2));
  ASSERT_FALSE(tuple2.greater(tuple1));

  // tuple1 vs tuple3
  ASSERT_TRUE(tuple3.greater(tuple1));
  ASSERT_FALSE(tuple1.greater(tuple3));
}

TYPED_TEST(TupleUnit, GreaterEqual)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple3{2, 3, 4, 5, 6};

  // tuple1 vs tuple1
  ASSERT_TRUE(tuple1.greater_equal(tuple1));

  // tuple1 vs tuple2
  ASSERT_TRUE(tuple1.greater_equal(tuple2));
  ASSERT_TRUE(tuple2.greater_equal(tuple1));

  // tuple1 vs tuple3
  ASSERT_TRUE(tuple3.greater_equal(tuple1));
  ASSERT_FALSE(tuple1.greater_equal(tuple3));
}

TYPED_TEST(TupleUnit, GetItem)
{
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5};
  ASSERT_EQ(tuple.at(0), 1);
  ASSERT_EQ(tuple[4], 5);
  ASSERT_EQ(tuple.at(2), tuple[2]);
}

TYPED_TEST(TupleUnit, AddComputation)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{2, 4, 6, 8, 15};
  const legate::tuple<typename TypeParam::value_type> tuple3{};

  ASSERT_EQ(tuple1 + tuple2, (legate::tuple<typename TypeParam::value_type>{3, 6, 9, 12, 20}));
  ASSERT_EQ(tuple1 + tuple1, (legate::tuple<typename TypeParam::value_type>{2, 4, 6, 8, 10}));
  ASSERT_EQ(tuple3 + tuple3, (legate::tuple<typename TypeParam::value_type>{}));
}

TYPED_TEST(TupleUnit, SubComputation)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{2, 4, 6, 8, 15};
  const legate::tuple<typename TypeParam::value_type> tuple3{};

  ASSERT_EQ(tuple2 - tuple1, (legate::tuple<typename TypeParam::value_type>{1, 2, 3, 4, 10}));
  ASSERT_EQ(tuple1 - tuple1, (legate::tuple<typename TypeParam::value_type>{0, 0, 0, 0, 0}));
  ASSERT_EQ(tuple3 - tuple3, (legate::tuple<typename TypeParam::value_type>{}));
}

TYPED_TEST(TupleUnit, MultiplyComputation)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{2, 4, 6, 8, 15};
  const legate::tuple<typename TypeParam::value_type> tuple3{};

  ASSERT_EQ(tuple2 * tuple1, (legate::tuple<typename TypeParam::value_type>{2, 8, 18, 32, 75}));
  ASSERT_EQ(tuple1 * tuple1, (legate::tuple<typename TypeParam::value_type>{1, 4, 9, 16, 25}));
  ASSERT_EQ(tuple3 * tuple3, (legate::tuple<typename TypeParam::value_type>{}));
}

TYPED_TEST(TupleUnit, DivideComputation)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{2, 4, 6, 8, 15};
  const legate::tuple<typename TypeParam::value_type> tuple3{};

  ASSERT_EQ(tuple2 / tuple1, (legate::tuple<typename TypeParam::value_type>{2, 2, 2, 2, 3}));
  ASSERT_EQ(tuple1 / tuple1, (legate::tuple<typename TypeParam::value_type>{1, 1, 1, 1, 1}));
  ASSERT_EQ(tuple3 / tuple3, (legate::tuple<typename TypeParam::value_type>{}));
}

TYPED_TEST(TupleUnit, Insert)
{
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3};
  const legate::tuple<typename TypeParam::value_type> tuple_empty{};

  const auto tuple1 = tuple.insert(2, 10);
  ASSERT_EQ(tuple1, (legate::tuple<typename TypeParam::value_type>{1, 2, 10, 3}));

  const auto tuple2 = tuple_empty.insert(0, 10);
  ASSERT_EQ(tuple2, (legate::tuple<typename TypeParam::value_type>{10}));
}

TYPED_TEST(TupleUnit, InsertInplace)
{
  legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3};
  legate::tuple<typename TypeParam::value_type> tuple_empty{};

  tuple.insert_inplace(3, 10);
  ASSERT_EQ(tuple, (legate::tuple<typename TypeParam::value_type>{1, 2, 3, 10}));

  tuple_empty.insert_inplace(0, 10);
  ASSERT_EQ(tuple_empty, (legate::tuple<typename TypeParam::value_type>{10}));
}

TYPED_TEST(TupleUnit, Append)
{
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3};
  const legate::tuple<typename TypeParam::value_type> tuple_empty{};

  const auto tuple1 = tuple.append(10);
  ASSERT_EQ(tuple1, (legate::tuple<typename TypeParam::value_type>{1, 2, 3, 10}));

  const auto tuple2 = tuple_empty.append(10);
  ASSERT_EQ(tuple2, (legate::tuple<typename TypeParam::value_type>{10}));
}

TYPED_TEST(TupleUnit, AppendInplace)
{
  legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3};
  legate::tuple<typename TypeParam::value_type> tuple_empty{};

  tuple.append_inplace(10);
  ASSERT_EQ(tuple, (legate::tuple<typename TypeParam::value_type>{1, 2, 3, 10}));

  tuple_empty.append_inplace(10);
  ASSERT_EQ(tuple_empty, (legate::tuple<typename TypeParam::value_type>{10}));
}

TYPED_TEST(TupleUnit, Remove)
{
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5};

  const auto tuple1 = tuple.remove(0);
  ASSERT_EQ(tuple1, (legate::tuple<typename TypeParam::value_type>{2, 3, 4, 5}));

  const auto tuple2 = tuple.remove(4);
  ASSERT_EQ(tuple2, (legate::tuple<typename TypeParam::value_type>{1, 2, 3, 4}));
}

TYPED_TEST(TupleUnit, RemoveInplace)
{
  legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5};

  tuple.remove_inplace(0);
  ASSERT_EQ(tuple, (legate::tuple<typename TypeParam::value_type>{2, 3, 4, 5}));

  tuple.remove_inplace(3);
  ASSERT_EQ(tuple, (legate::tuple<typename TypeParam::value_type>{2, 3, 4}));
}

TYPED_TEST(TupleUnit, Update)
{
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5};
  const auto tuple1 = tuple.update(0, 10);
  ASSERT_EQ(tuple1, (legate::tuple<typename TypeParam::value_type>{10, 2, 3, 4, 5}));

  const auto tuple2 = tuple.update(4, 10);
  ASSERT_EQ(tuple2, (legate::tuple<typename TypeParam::value_type>{1, 2, 3, 4, 10}));
}

TYPED_TEST(TupleUnit, Map)
{
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5, 6};
  const std::vector<std::int32_t> mapping1{0, 2, 4, 1, 5, 3};
  ASSERT_EQ(tuple.map(mapping1), (legate::tuple<typename TypeParam::value_type>{1, 3, 5, 2, 6, 4}));

  const legate::tuple<typename TypeParam::value_type> tuple_empty{};
  const std::vector<std::int32_t> mapping2{};
  ASSERT_EQ(tuple_empty.map(mapping2), (legate::tuple<typename TypeParam::value_type>{}));
}

TYPED_TEST(TupleUnit, MapInplace)
{
  legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5, 6};
  std::vector<std::int32_t> mapping1{5, 3, 1, 0, 4, 2};
  tuple.map_inplace(mapping1);
  ASSERT_EQ(tuple, (legate::tuple<typename TypeParam::value_type>{6, 4, 2, 1, 5, 3}));
  ASSERT_EQ(mapping1, (std::vector<std::int32_t>{0, 1, 2, 3, 4, 5}));

  legate::tuple<typename TypeParam::value_type> tuple_empty{};
  std::vector<std::int32_t> mapping2{};
  tuple_empty.map_inplace(mapping2);
  ASSERT_EQ(tuple_empty, (legate::tuple<typename TypeParam::value_type>{}));
  ASSERT_EQ(mapping2, (std::vector<std::int32_t>{}));
}

TYPED_TEST(TupleUnit, Reduce)
{
  auto compare = [](auto&& v1, auto&& v2) { return v2 > v1 ? v2 : v1; };
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple_empty{};

  const typename TypeParam::value_type init1{2};
  ASSERT_EQ(tuple.reduce(compare, init1), 5);

  const typename TypeParam::value_type init2{6};
  ASSERT_EQ(tuple.reduce(compare, init2), 6);

  const typename TypeParam::value_type init3{3};
  ASSERT_EQ(tuple_empty.reduce(compare, init3), 3);
}

TYPED_TEST(TupleUnit, Sum)
{
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple_empty{};

  ASSERT_EQ(tuple.sum(), static_cast<typename TypeParam::value_type>(15));
  ASSERT_EQ(tuple_empty.sum(), static_cast<typename TypeParam::value_type>(0));
}

TYPED_TEST(TupleUnit, Volume)
{
  const legate::tuple<typename TypeParam::value_type> tuple{1, 2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple_empty{};

  ASSERT_EQ(tuple.volume(), static_cast<typename TypeParam::value_type>(120));
  ASSERT_EQ(tuple_empty.volume(), static_cast<typename TypeParam::value_type>(1));
}

TYPED_TEST(TupleUnit, All)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{2, 0, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple3{2, 3, 4, 5, 1};
  const legate::tuple<typename TypeParam::value_type> tuple4{};

  auto greater_than_one = [](auto&& v) { return v > 1; };
  ASSERT_TRUE(tuple1.all(greater_than_one));
  ASSERT_FALSE(tuple2.all(greater_than_one));
  ASSERT_FALSE(tuple3.all(greater_than_one));
  ASSERT_TRUE(tuple4.all(greater_than_one));

  ASSERT_TRUE(tuple1.all());
  ASSERT_FALSE(tuple2.all());
  ASSERT_TRUE(tuple3.all());
  ASSERT_TRUE(tuple4.all());
}

TYPED_TEST(TupleUnit, Any)
{
  const legate::tuple<typename TypeParam::value_type> tuple1{2, 3, 4, 5};
  const legate::tuple<typename TypeParam::value_type> tuple2{0, 0, 0, 0, 0};
  const legate::tuple<typename TypeParam::value_type> tuple3{1, 2, 3, 4, 5, 0};
  const legate::tuple<typename TypeParam::value_type> tuple4{};

  auto less_than_one = [](auto&& v) { return v < 1; };
  ASSERT_FALSE(tuple1.any(less_than_one));
  ASSERT_TRUE(tuple2.any(less_than_one));
  ASSERT_TRUE(tuple3.any(less_than_one));
  ASSERT_FALSE(tuple4.any(less_than_one));

  ASSERT_TRUE(tuple1.any());
  ASSERT_FALSE(tuple2.any());
  ASSERT_TRUE(tuple3.any());
  ASSERT_FALSE(tuple4.any());
}

TEST(TupleUnit, ModComputation)
{
  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{2, 5, 6, 10, 8};
  const legate::tuple<std::int32_t> tuple3{};

  const legate::tuple<std::int32_t> tuple_mod{0, 0, 0, 0, 0};
  ASSERT_EQ(tuple2 % tuple1, (legate::tuple<int>{0, 1, 0, 2, 3}));
  ASSERT_EQ(tuple1 % tuple1, (legate::tuple<int>{0, 0, 0, 0, 0}));
  ASSERT_EQ(tuple3 % tuple3, (legate::tuple<int>{}));
}

TEST(TupleUnitNegative, Less)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(static_cast<void>(tuple1.less(tuple2)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple2.less(tuple1)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple1.less(tuple3)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple3.less(tuple1)), std::invalid_argument);
}

TEST(TupleUnitNegative, LessEqual)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(static_cast<void>(tuple1.less_equal(tuple2)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple2.less_equal(tuple1)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple1.less_equal(tuple3)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple3.less_equal(tuple1)), std::invalid_argument);
}

TEST(TupleUnitNegative, Greater)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(static_cast<void>(tuple1.greater(tuple2)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple2.greater(tuple1)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple1.greater(tuple3)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple3.greater(tuple1)), std::invalid_argument);
}

TEST(TupleUnitNegative, GreaterEqual)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(static_cast<void>(tuple1.greater_equal(tuple2)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple2.greater_equal(tuple1)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple1.greater_equal(tuple3)), std::invalid_argument);
  ASSERT_THROW(static_cast<void>(tuple3.greater_equal(tuple1)), std::invalid_argument);
}

TEST(TupleUnitNegative, GetItem)
{
  const legate::tuple<std::int32_t> tuple{1, 2, 3, 4, 5};
  ASSERT_THROW(static_cast<void>(tuple.at(5)), std::out_of_range);
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    ASSERT_THROW(static_cast<void>(tuple[5]), std::out_of_range);
  }
}

TEST(TupleUnitNegative, AddComputation)
{
  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(tuple1 + tuple2, std::invalid_argument);
  ASSERT_THROW(tuple1 + tuple3, std::invalid_argument);
}

TEST(TupleUnitNegative, SubComputation)
{
  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(tuple1 - tuple2, std::invalid_argument);
  ASSERT_THROW(tuple1 - tuple3, std::invalid_argument);
}

TEST(TupleUnitNegative, MultiplyComputation)
{
  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(tuple1 * tuple2, std::invalid_argument);
  ASSERT_THROW(tuple1 * tuple3, std::invalid_argument);
}

TEST(TupleUnitNegative, DivideComputation)
{
  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(tuple1 / tuple2, std::invalid_argument);
  ASSERT_THROW(tuple1 / tuple3, std::invalid_argument);
}

TEST(TupleUnitNegative, ModComputation)
{
  const legate::tuple<std::int32_t> tuple1{1, 2, 3, 4, 5};
  const legate::tuple<std::int32_t> tuple2{1, 2, 3, 4, 5, 6};
  const legate::tuple<std::int32_t> tuple3{};

  ASSERT_THROW(tuple1 % tuple2, std::invalid_argument);
  ASSERT_THROW(tuple1 % tuple3, std::invalid_argument);
}

TEST(TupleUnitNegative, Map)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  const legate::tuple<std::int32_t> tuple{1, 2, 3, 4, 5, 6};

  const std::vector<std::int32_t> mapping1{0, 2, 4};
  ASSERT_THROW(static_cast<void>(tuple.map(mapping1)), std::out_of_range);

  const std::vector<std::int32_t> mapping2{-1, 1, 2, 3, 4, 5};
  ASSERT_THROW(static_cast<void>(tuple.map(mapping2)), std::out_of_range);

  const std::vector<std::int32_t> mapping3{1, 2, 3, 4, 5, 6};
  ASSERT_THROW(static_cast<void>(tuple.map(mapping3)), std::out_of_range);

  const std::vector<std::int32_t> mapping4{1, 1, 2, 3, 4, 5};
  ASSERT_THROW(static_cast<void>(tuple.map(mapping4)), std::invalid_argument);
}

TEST(TupleUnitNegative, MapInplace)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  legate::tuple<std::int32_t> tuple{1, 2, 3, 4, 5, 6};

  std::vector<std::int32_t> mapping1{0, 2, 4};
  ASSERT_THROW(tuple.map_inplace(mapping1), std::out_of_range);

  std::vector<std::int32_t> mapping2{-1, 1, 2, 3, 4, 5};
  ASSERT_THROW(tuple.map_inplace(mapping2), std::out_of_range);

  std::vector<std::int32_t> mapping3{1, 2, 3, 4, 5, 6};
  ASSERT_THROW(tuple.map_inplace(mapping3), std::out_of_range);

  std::vector<std::int32_t> mapping4{1, 1, 2, 3, 4, 5};
  ASSERT_THROW(tuple.map_inplace(mapping4), std::invalid_argument);
}

TEST(TupleUnitNegative, Insert)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "The functions don't throw if LEGATE_USE_DEBUG is not defined";
  }

  const legate::tuple<std::int32_t> tuple{1, 2, 3};
  const legate::tuple<std::int32_t> tuple_empty{};

  ASSERT_THROW(static_cast<void>(tuple.insert(4, 10)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple.insert(-1, 10)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple_empty.insert(-1, 10)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple_empty.insert(1, 10)), std::out_of_range);
}

TEST(TupleUnitNegative, InsertInplace)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "The functions don't throw if LEGATE_USE_DEBUG is not defined";
  }

  legate::tuple<std::int32_t> tuple{1, 2, 3};
  legate::tuple<std::int32_t> tuple_empty{};

  ASSERT_THROW(tuple.insert_inplace(4, 10), std::out_of_range);
  ASSERT_THROW(tuple_empty.insert_inplace(1, 10), std::out_of_range);
}

TEST(TupleUnitNegative, Remove)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "The functions don't throw if LEGATE_USE_DEBUG is not defined";
  }

  const legate::tuple<std::int32_t> tuple{1, 2, 3};
  const legate::tuple<std::int32_t> tuple_empty{};

  ASSERT_THROW(static_cast<void>(tuple.remove(4)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple.remove(-1)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple_empty.remove(0)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple_empty.remove(-1)), std::out_of_range);
}

LEGATE_PRAGMA_PUSH();
LEGATE_PRAGMA_GCC_IGNORE("-Wstringop-overflow");

TEST(TupleUnitNegative, RemoveInplace)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  legate::tuple<std::int32_t> tuple{1, 2, 3};
  legate::tuple<std::int32_t> tuple_empty{};

  ASSERT_THROW(tuple.remove_inplace(4), std::out_of_range);
  ASSERT_THROW(tuple.remove_inplace(-1), std::out_of_range);
  ASSERT_THROW(tuple_empty.remove_inplace(0), std::out_of_range);
  ASSERT_THROW(tuple_empty.remove_inplace(-1), std::out_of_range);
}

LEGATE_PRAGMA_POP();

TEST(TupleUnitNegative, Update)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_DEBUG is defined";
  }

  const legate::tuple<std::int32_t> tuple{1, 2, 3};
  const legate::tuple<std::int32_t> tuple_empty{};

  ASSERT_THROW(static_cast<void>(tuple.update(4, 10)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple.update(-1, 10)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple_empty.update(0, 10)), std::out_of_range);
  ASSERT_THROW(static_cast<void>(tuple_empty.update(-1, 10)), std::out_of_range);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace tuple_test
