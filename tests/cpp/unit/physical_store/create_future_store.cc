/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace create_future_physical_store_test {

namespace {

constexpr float FLOAT_VALUE   = 30.0F;
constexpr double DOUBLE_VALUE = 1000.0;

using CreateFuturePhysicalStoreUnit = DefaultFixture;

class CreateFutureStoreTest : public CreateFuturePhysicalStoreUnit,
                              public ::testing::WithParamInterface<legate::Scalar> {};

class NegativeCreateFutureStoreTest
  : public CreateFuturePhysicalStoreUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Scalar, legate::Shape>> {};

INSTANTIATE_TEST_SUITE_P(
  CreateFuturePhysicalStoreUnit,
  CreateFutureStoreTest,
  ::testing::Values(legate::Scalar{true, legate::bool_()},
                    legate::Scalar{static_cast<std::int8_t>(1), legate::int8()},
                    legate::Scalar{static_cast<std::int16_t>(2), legate::int16()},
                    legate::Scalar{static_cast<std::int32_t>(3), legate::int32()},
                    legate::Scalar{static_cast<std::int64_t>(4), legate::int64()},
                    legate::Scalar{static_cast<std::uint8_t>(5), legate::uint8()},
                    legate::Scalar{static_cast<std::uint16_t>(6), legate::uint16()},
                    legate::Scalar{static_cast<std::uint32_t>(7), legate::uint32()},
                    legate::Scalar{static_cast<std::uint64_t>(8), legate::uint64()},
                    legate::Scalar{static_cast<legate::Half>(FLOAT_VALUE), legate::float16()},
                    legate::Scalar{FLOAT_VALUE, legate::float32()},
                    legate::Scalar{DOUBLE_VALUE, legate::float64()},
                    legate::Scalar{legate::Complex<float>{FLOAT_VALUE, FLOAT_VALUE}},
                    legate::Scalar{legate::Complex<double>{FLOAT_VALUE, FLOAT_VALUE},
                                   legate::complex128()}));

INSTANTIATE_TEST_SUITE_P(
  CreateFuturePhysicalStoreUnit,
  NegativeCreateFutureStoreTest,
  ::testing::Values(std::make_tuple(legate::Scalar{"hello"},
                                    legate::Shape{1}) /* type of scalar has variable size  */,
                    std::make_tuple(legate::Scalar{1},
                                    legate::Shape{2}) /* shape.volume() != 1 */));

class FutureStoreFn {
 public:
  template <legate::Type::Code CODE>
  void operator()(const legate::Scalar& scalar) const
  {
    auto runtime               = legate::Runtime::get_runtime();
    auto logical_store         = runtime->create_store(scalar);
    auto store                 = logical_store.get_physical_store();
    constexpr std::int32_t DIM = 1;
    using T                    = legate::type_of_t<CODE>;

    ASSERT_TRUE(store.is_future());
    ASSERT_FALSE(store.is_unbound_store());
    ASSERT_EQ(store.dim(), DIM);
    ASSERT_TRUE(store.valid());
    ASSERT_EQ(store.type().code(), legate::type_code_of_v<T>);
    ASSERT_EQ(store.code(), legate::type_code_of_v<T>);
    ASSERT_FALSE(store.transformed());

    auto expect_rect = legate::Rect<DIM>{0, 0};
    auto domain      = store.domain();
    auto actual_rect = domain.bounds<DIM, std::size_t>();

    ASSERT_EQ(store.shape<DIM>(), expect_rect);
    ASSERT_EQ(domain.get_dim(), DIM);
    ASSERT_EQ(actual_rect, expect_rect);

    // Specific API for future store
    ASSERT_EQ(store.scalar<T>(), scalar.value<T>());
  }
};

}  // namespace

TEST_P(CreateFutureStoreTest, Basic)
{
  auto scalar = GetParam();

  legate::type_dispatch(scalar.type().code(), FutureStoreFn{}, scalar);
}

TEST_F(CreateFuturePhysicalStoreUnit, StoreCreationLike)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{FLOAT_VALUE});
  auto store         = logical_store.get_physical_store();
  // testing for constructor of PhysicalStore
  const legate::PhysicalStore other1{store};  // NOLINT(performance-unnecessary-copy-initialization)
  static constexpr auto DIM = 1;

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

TEST_F(CreateFuturePhysicalStoreUnit, Assignment)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{FLOAT_VALUE});
  auto store         = logical_store.get_physical_store();
  // tesing for operator=
  const auto other1         = store;  // NOLINT(performance-unnecessary-copy-initialization)
  static constexpr auto DIM = 1;

  ASSERT_EQ(other1.dim(), store.dim());
  ASSERT_EQ(other1.type().code(), store.type().code());
  ASSERT_EQ(other1.shape<DIM>(), store.shape<DIM>());

  const auto other2 = logical_store.get_physical_store();

  ASSERT_EQ(other2.dim(), store.dim());
  ASSERT_EQ(other2.type().code(), store.type().code());
  ASSERT_EQ(other2.shape<DIM>(), store.shape<DIM>());
}

TEST_F(CreateFuturePhysicalStoreUnit, InvalidDim)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{1});
  auto store         = logical_store.get_physical_store();

  ASSERT_THROW(static_cast<void>(store.shape<2>()), std::invalid_argument);
}

TEST_F(CreateFuturePhysicalStoreUnit, InvalidBind)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store(legate::Scalar{1});
  auto store         = logical_store.get_physical_store();

  // Specific APIs for bound/unbound stores
  ASSERT_THROW(
    static_cast<void>(store.create_output_buffer<std::uint64_t>(legate::Point<1>::ONES())),
    std::invalid_argument);
  ASSERT_THROW(store.bind_empty_data(), std::invalid_argument);
}

TEST_P(NegativeCreateFutureStoreTest, InvalidCreate)
{
  const auto [scalar, shape] = GetParam();
  auto runtime               = legate::Runtime::get_runtime();

  ASSERT_THROW(static_cast<void>(runtime->create_store(scalar, shape)), std::invalid_argument);
}

}  // namespace create_future_physical_store_test
