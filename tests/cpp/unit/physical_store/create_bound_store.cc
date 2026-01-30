/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/buffer.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/physical_stores/region_physical_store.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/mapping/mapping.h>
#include <legate/redop/redop.h>
#include <legate/type/detail/types.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <utilities/utilities.h>
#include <utility>  // std::as_const

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
    using T = legate::type_of_t<CODE>;

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
    static_cast<void>(store.span_read_accessor<T, DIM>());
    static_cast<void>(store.span_write_accessor<T, DIM>());
    static_cast<void>(store.span_read_write_accessor<T, DIM>());
    static_cast<void>(store.read_accessor<T, DIM>());
    static_cast<void>(store.write_accessor<T, DIM>());
    static_cast<void>(store.read_write_accessor<T, DIM>());
    static_cast<void>(store.reduce_accessor<legate::SumReduction<T>, false, DIM>());
    ASSERT_TRUE(store.is_writable());
    ASSERT_TRUE(store.is_reducible());
    const legate::Rect<DIM> expect_rect{legate::Point<DIM>{lo.data()},
                                        legate::Point<DIM>{hi.data()}};

    ASSERT_EQ(store.shape<DIM>(), expect_rect);
    auto domain      = store.domain();
    auto actual_rect = domain.bounds<DIM, std::size_t>();

    ASSERT_EQ(domain.get_dim(), DIM);
    ASSERT_EQ(actual_rect, expect_rect);
  }
};

class DummyPhysicalStore final : public legate::detail::PhysicalStore {
 public:
  DummyPhysicalStore(std::int32_t dim,
                     legate::InternalSharedPtr<legate::detail::Type> type,
                     legate::GlobalRedopID redop_id,
                     legate::InternalSharedPtr<legate::detail::TransformStack> transform)
    : legate::detail::PhysicalStore{dim,
                                    std::move(type),
                                    redop_id,
                                    std::move(transform),
                                    /*readable=*/true,
                                    /*writable=*/true,
                                    /*reducible=*/false}
  {
  }

  [[nodiscard]] bool valid() const override { return true; }

  [[nodiscard]] legate::Domain domain() const override { return {}; }

  [[nodiscard]] legate::InlineAllocation get_inline_allocation() const override { return {}; }

  [[nodiscard]] legate::mapping::StoreTarget target() const override
  {
    return legate::mapping::StoreTarget::SYSMEM;
  }

  [[nodiscard]] bool is_partitioned() const override { return false; }
};

struct UnregisteredType {
  std::uint64_t value{};
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

  // 0-dim store can be accessed as 1D (special case allowed)
  auto rect1 = store.shape<1>();
  ASSERT_EQ(rect1, (legate::Rect<1>{0, 0}));

  // 0-dim store cannot be accessed as 2D or higher
  ASSERT_THAT(
    [&] { static_cast<void>(store.shape<2>()); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("invalid to retrieve a")));

  // 0-dim store can create 1D accessor (special case allowed)
  ASSERT_NO_THROW(static_cast<void>(store.read_accessor<std::int64_t, 1>()));

  // 0-dim store cannot create 2D accessor
  ASSERT_THAT([&] { static_cast<void>(store.read_accessor<std::int64_t, 2>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("invalid to retrieve a 2-D rect")));
}

TEST_F(CreateBoundPhysicalStoreUnit, OptimizeScalar)
{
  auto runtime        = legate::Runtime::get_runtime();
  auto logical_store1 = runtime->create_store({1}, legate::int64(), /*optimize_scalar=*/true);
  auto store1         = logical_store1.get_physical_store();

  ASSERT_TRUE(store1.is_future());

  auto logical_store2 = runtime->create_store({1, 2}, legate::int64(), /*optimize_scalar=*/true);
  auto store2         = logical_store2.get_physical_store();

  ASSERT_FALSE(store2.is_future());

  auto logical_store3 = runtime->create_store({1}, legate::int64(), /*optimize_scalar=*/false);
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

  // Cover const version of as_region_store()
  const auto& region_store = std::as_const(*store.impl()).as_region_store();

  ASSERT_TRUE(region_store.valid());
}

TEST_F(CreateBoundPhysicalStoreUnit, DetailPhysicalStoreDestructor)
{
  auto type      = legate::detail::int32();
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();
  auto* store    = new DummyPhysicalStore{
    /*dim=*/1, std::move(type), legate::GlobalRedopID{0}, std::move(transform)};

  const legate::detail::PhysicalStore* base = store;
  delete base;
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

  ASSERT_THAT(
    [&] { static_cast<void>(store.shape<INVALID_DIM>()); },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr("invalid to retrieve a")));
}

TEST_F(CreateBoundPhysicalStoreUnit, InvalidBind)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2}, legate::int64());
  auto store         = logical_store.get_physical_store();

  ASSERT_THAT(
    [&] { static_cast<void>(store.create_output_buffer<std::int64_t>(legate::Point<1>::ONES())); },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Store isn't an unbound store")));
  ASSERT_THAT([&] { store.bind_empty_data(); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Empty data can only be bound to unbound stores")));
}

TEST_F(CreateBoundPhysicalStoreUnit, BindTaskLocalBufferInvalid)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2}, legate::int32());
  auto store         = logical_store.get_physical_store();
  std::array<std::uint64_t, 1> bounds{1};
  legate::TaskLocalBuffer buffer_same{legate::int32(), bounds};
  legate::TaskLocalBuffer buffer_other{legate::int64(), bounds};
  legate::DomainPoint extents{legate::Point<1>{1}};

  ASSERT_THAT([&] { store.bind_data(buffer_other, extents, true); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("types are not compatible")));
  ASSERT_THAT([&] { store.bind_data(buffer_same, extents, true); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Data can only be bound to unbound stores")));
}

TEST_F(CreateBoundPhysicalStoreUnit, BindUntypedBufferInvalid)
{
  auto runtime                = legate::Runtime::get_runtime();
  auto logical_store          = runtime->create_store({2}, legate::int32());
  auto store                  = logical_store.get_physical_store();
  constexpr auto num_elements = 4;
  auto buffer                 = legate::create_buffer<std::int8_t, 1>(num_elements);

  ASSERT_THAT([&] { store.bind_untyped_data(buffer, legate::Point<1>{num_elements}); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Untyped data can only be bound to unbound stores")));
}

TEST_F(CreateBoundPhysicalStoreUnit, InvalidScalar)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2}, legate::int64());
  auto store         = logical_store.get_physical_store();

  ASSERT_THAT([&] { static_cast<void>(store.scalar<std::int64_t>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Store isn't a scalar store")));
}

TEST_F(CreateBoundPhysicalStoreUnit, InvalidAccessorTypeSize)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto logical_store = runtime->create_store({2}, legate::int32());
  auto store         = logical_store.get_physical_store();

  ASSERT_THAT([&] { static_cast<void>(store.read_accessor<UnregisteredType, 1, true>()); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Type size mismatch: store type")));
}

}  // namespace create_bound_physical_store_test
