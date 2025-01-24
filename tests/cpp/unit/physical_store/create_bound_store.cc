/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace create_bound_physical_store_test {

namespace {

using CreateBoundPhysicalStoreUnit = DefaultFixture;

class CreateBoundStoreTest
  : public CreateBoundPhysicalStoreUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Shape,
                                                    legate::Type,
                                                    std::vector<legate::coord_t>,
                                                    std::vector<legate::coord_t>>> {};

// NOLINTBEGIN(readability-magic-numbers)

std::vector<
  std::
    tuple<legate::Shape, legate::Type, std::vector<legate::coord_t>, std::vector<legate::coord_t>>>
bound_store_cases()
{
  std::vector<std::tuple<legate::Shape,
                         legate::Type,
                         std::vector<legate::coord_t>,
                         std::vector<legate::coord_t>>>
    cases = {{legate::Shape{5},
              legate::bool_(),
              std::vector<legate::coord_t>({0}),
              std::vector<legate::coord_t>({4})},
             {legate::Shape{static_cast<std::uint64_t>(-2), 1},
              legate::int64(),
              std::vector<legate::coord_t>({0, 0}),
              std::vector<legate::coord_t>({-3, 0})},
             {legate::Shape{100, 10, 1},
              legate::float16(),
              std::vector<legate::coord_t>({0, 0, 0}),
              std::vector<legate::coord_t>({99, 9, 0})},
             {legate::Shape{7, 100, 8, 100},
              legate::complex128(),
              std::vector<legate::coord_t>({0, 0, 0, 0}),
              std::vector<legate::coord_t>({6, 99, 7, 99})}};

#if LEGATE_MAX_DIM >= 5
  cases.emplace_back(legate::Shape{20, 6, 4, 10, 50},
                     legate::uint16(),
                     std::vector<legate::coord_t>({0, 0, 0, 0, 0}),
                     std::vector<legate::coord_t>({19, 5, 3, 9, 49}));
#endif
#if LEGATE_MAX_DIM >= 6
  cases.emplace_back(legate::Shape{1, 2, 3, 4, 5, 6},
                     legate::float64(),
                     std::vector<legate::coord_t>({0, 0, 0, 0, 0, 0}),
                     std::vector<legate::coord_t>({0, 1, 2, 3, 4, 5}));
#endif
#if LEGATE_MAX_DIM >= 7
  cases.emplace_back(legate::Shape{7, 6, 5, 4, 3, 2, 1},
                     legate::complex64(),
                     std::vector<legate::coord_t>({0, 0, 0, 0, 0, 0, 0}),
                     std::vector<legate::coord_t>({6, 5, 4, 3, 2, 1, 0}));
#endif

  return cases;
}

// NOLINTEND(readability-magic-numbers)

INSTANTIATE_TEST_SUITE_P(CreateBoundPhysicalStoreUnit,
                         CreateBoundStoreTest,
                         ::testing::ValuesIn(bound_store_cases()));

class BoundStoreFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::Shape& shape,
                  const legate::Type& type,
                  const std::vector<legate::coord_t>& lo,
                  const std::vector<legate::coord_t>& hi) const
  {
    auto runtime       = legate::Runtime::get_runtime();
    auto logical_store = runtime->create_store(shape, type);
    auto store         = logical_store.get_physical_store();

    ASSERT_FALSE(store.is_future());
    ASSERT_FALSE(store.is_unbound_store());
    ASSERT_EQ(store.dim(), DIM);
    ASSERT_TRUE(store.valid());
    ASSERT_EQ(store.type().code(), CODE);
    ASSERT_EQ(store.code(), CODE);
    ASSERT_FALSE(store.transformed());
    const legate::Rect<DIM> expect_rect{legate::Point<DIM>{lo.data()},
                                        legate::Point<DIM>{hi.data()}};

    ASSERT_EQ(store.shape<DIM>(), expect_rect);
    auto domain      = store.domain();
    auto actual_rect = domain.bounds<DIM, std::size_t>();

    ASSERT_EQ(domain.get_dim(), DIM);
    ASSERT_EQ(actual_rect, expect_rect);
  }
};

}  // namespace

TEST_P(CreateBoundStoreTest, Basic)
{
  auto [shape, type, lo, hi] = GetParam();

  legate::double_dispatch(
    static_cast<int>(shape.ndim()), type.code(), BoundStoreFn{}, shape, type, lo, hi);
}

TEST_F(CreateBoundPhysicalStoreUnit, EmptyShape)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({}, legate::int64());
  auto store         = logical_store.get_physical_store();

  ASSERT_EQ(store.dim(), 0);
}

TEST_F(CreateBoundPhysicalStoreUnit, OptimizeScalar)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_store1 = runtime->create_store({1}, legate::int64(), true);
  auto store1         = logical_store1.get_physical_store();

  ASSERT_TRUE(store1.is_future());

  auto logical_store2 = runtime->create_store({1, 2}, legate::int64(), true);
  auto store2         = logical_store2.get_physical_store();

  ASSERT_FALSE(store2.is_future());

  auto logical_store3 = runtime->create_store({1}, legate::int64(), false);
  auto store3         = logical_store3.get_physical_store();

  ASSERT_FALSE(store3.is_future());
}

TEST_F(CreateBoundPhysicalStoreUnit, Valid)
{
  auto runtime                  = legate::Runtime::get_runtime();
  static constexpr auto EXTENTS = 5;
  auto logical_store = runtime->create_store(legate::Shape{0, EXTENTS}, legate::uint64());
  auto store         = logical_store.get_physical_store();

  ASSERT_TRUE(store.valid());

  auto moved_store = std::move(store);

  // to test the branch where impl() is nullptr
  ASSERT_FALSE(store.valid());  // NOLINT(bugprone-use-after-move,clang-analyzer-cplusplus.Move)
  ASSERT_TRUE(moved_store.valid());
}

TEST_F(CreateBoundPhysicalStoreUnit, StoreCreationLike)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2, 3}, legate::int64());
  auto store         = logical_store.get_physical_store();

  // testing for constructor of PhysicalStore
  const legate::PhysicalStore other1{store};  // NOLINT(performance-unnecessary-copy-initialization)
  static constexpr auto DIM = 2;

  ASSERT_EQ(other1.dim(), store.dim());
  ASSERT_EQ(other1.type().code(), store.type().code());
  ASSERT_EQ(other1.shape<DIM>(), store.shape<DIM>());
  // testing for constructor of PhysicalStore
  const legate::PhysicalStore other2{
    logical_store.get_physical_store()};  // NOLINT(performance-unnecessary-copy-initialization)

  ASSERT_EQ(other2.dim(), store.dim());
  ASSERT_EQ(other2.type().code(), store.type().code());
  ASSERT_EQ(other2.shape<DIM>(), store.shape<DIM>());
}

TEST_F(CreateBoundPhysicalStoreUnit, Assignment)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2, 3}, legate::int64());
  auto store         = logical_store.get_physical_store();
  // tesing for operator=
  const auto other1         = store;  // NOLINT(performance-unnecessary-copy-initialization)
  static constexpr auto DIM = 2;

  ASSERT_EQ(other1.dim(), store.dim());
  ASSERT_EQ(other1.type().code(), store.type().code());
  ASSERT_EQ(other1.shape<DIM>(), store.shape<DIM>());

  const auto other2 = logical_store.get_physical_store();

  ASSERT_EQ(other2.dim(), store.dim());
  ASSERT_EQ(other2.type().code(), store.type().code());
  ASSERT_EQ(other2.shape<DIM>(), store.shape<DIM>());
}

TEST_F(CreateBoundPhysicalStoreUnit, InvalidDim)
{
  auto runtime                       = legate::Runtime::get_runtime();
  auto logical_store                 = runtime->create_store({2}, legate::int64());
  auto store                         = logical_store.get_physical_store();
  constexpr std::int32_t INVALID_DIM = 2;

  ASSERT_THROW(static_cast<void>(store.shape<INVALID_DIM>()), std::invalid_argument);
}

TEST_F(CreateBoundPhysicalStoreUnit, InvalidBind)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2}, legate::int64());
  auto store         = logical_store.get_physical_store();

  ASSERT_THROW(
    static_cast<void>(store.create_output_buffer<std::int64_t>(legate::Point<1>::ONES())),
    std::invalid_argument);
  ASSERT_THROW(store.bind_empty_data(), std::invalid_argument);
}

TEST_F(CreateBoundPhysicalStoreUnit, InvalidScalar)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2}, legate::int64());
  auto store         = logical_store.get_physical_store();

  ASSERT_THROW(static_cast<void>(store.scalar<std::int64_t>()), std::invalid_argument);
}

}  // namespace create_bound_physical_store_test
