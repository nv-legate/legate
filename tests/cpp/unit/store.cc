#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <valarray>

#include "legate.h"

namespace unit {

void store_creation(int32_t argc, char** argv)
{
  // Bound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store({4, 4}, legate::LegateTypeCode::INT64_LT);
    EXPECT_FALSE(store.unbound());
    EXPECT_EQ(store.dim(), 2);
    EXPECT_EQ(store.extents(), (std::vector<size_t>{4, 4}));
    EXPECT_EQ(store.code(), legate::LegateTypeCode::INT64_LT);
    EXPECT_FALSE(store.transformed());
  }

  // Unbound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store(legate::LegateTypeCode::INT64_LT);
    EXPECT_TRUE(store.unbound());
    EXPECT_EQ(store.dim(), 1);
    EXPECT_EQ(store.code(), legate::LegateTypeCode::INT64_LT);
    EXPECT_FALSE(store.transformed());
    EXPECT_THROW(store.extents(), std::invalid_argument);
  }
}

void store_valid_transform(int32_t argc, char** argv)
{
  // Bound
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store({4, 3}, legate::LegateTypeCode::INT64_LT);

  auto promoted = store.promote(0, 5);
  EXPECT_EQ(promoted.extents(), (std::vector<size_t>{5, 4, 3}));
  EXPECT_TRUE(promoted.transformed());

  auto projected = store.project(0, 1);
  EXPECT_EQ(projected.extents(),
            (std::vector<size_t>{
              3,
            }));
  EXPECT_TRUE(projected.transformed());

  auto sliced = store.slice(1, std::slice(1, 3, 1));
  EXPECT_EQ(sliced.extents(), (std::vector<size_t>{4, 2}));
  EXPECT_TRUE(sliced.transformed());

  auto transposed = store.transpose({1, 0});
  EXPECT_EQ(transposed.extents(), (std::vector<size_t>{3, 4}));
  EXPECT_TRUE(transposed.transformed());

  auto delinearized = store.delinearize(0, {2, 2});
  EXPECT_EQ(delinearized.extents(), (std::vector<size_t>{2, 2, 3}));
  EXPECT_TRUE(delinearized.transformed());
}

void store_invalid_transform(int32_t argc, char** argv)
{
  // Bound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store({4, 3}, legate::LegateTypeCode::INT64_LT);

    EXPECT_THROW(store.promote(3, 5), std::invalid_argument);
    EXPECT_THROW(store.promote(-3, 5), std::invalid_argument);

    EXPECT_THROW(store.project(2, 1), std::invalid_argument);
    EXPECT_THROW(store.project(-3, 1), std::invalid_argument);
    EXPECT_THROW(store.project(0, 4), std::invalid_argument);

    EXPECT_THROW(store.slice(2, std::slice(1, 3, 1)), std::invalid_argument);
    EXPECT_THROW(store.slice(1, std::slice(1, 3, 2)), std::invalid_argument);
    EXPECT_THROW(store.slice(1, std::slice(1, 4, 1)), std::invalid_argument);

    EXPECT_THROW(store.transpose({
                   2,
                 }),
                 std::invalid_argument);
    EXPECT_THROW(store.transpose({0, 0}), std::invalid_argument);
    EXPECT_THROW(store.transpose({2, 0}), std::invalid_argument);

    EXPECT_THROW(store.delinearize(2, {2, 3}), std::invalid_argument);
    EXPECT_THROW(store.delinearize(0, {2, 3}), std::invalid_argument);
  }

  // Unbound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store(legate::LegateTypeCode::INT64_LT);
    EXPECT_THROW(store.promote(1, 1), std::invalid_argument);
  }
}

}  // namespace unit

TEST(Store, Creation)
{
  legate::initialize(0, NULL);
  legate::set_main_function(unit::store_creation);
  legate::start(0, NULL);
}

TEST(Store, Transform)
{
  legate::initialize(0, NULL);
  legate::set_main_function(unit::store_valid_transform);
  legate::start(0, NULL);
}

TEST(Store, InvalidTransform)
{
  legate::initialize(0, NULL);
  legate::set_main_function(unit::store_invalid_transform);
  legate::start(0, NULL);
}
