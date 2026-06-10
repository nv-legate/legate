/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/transform/project.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/detail/store.h>

#include <gtest/gtest.h>

#include <memory>
#include <utilities/mock_mapper.h>
#include <utilities/utilities.h>

namespace dim_ordering_unit {

namespace {

using legate::test::MockMapperRuntime;

using DimOrderingTest = DefaultFixture;

void check_dim_ordering(const legate::mapping::DimOrdering& order,
                        legate::mapping::DimOrdering::Kind kind,
                        const std::vector<std::int32_t>& dim)
{
  ASSERT_EQ(order.kind(), kind);
  ASSERT_EQ(order.dimensions(), dim);
}

void check_legion_dims(const legate::mapping::DimOrdering& order,
                       std::uint32_t ndim,
                       const std::vector<Legion::DimensionKind>& expected)
{
  ASSERT_EQ(order.impl()->generate_legion_dims(ndim), expected);
}

}  // namespace

TEST_F(DimOrderingTest, CreateDefault)
{
  const std::vector<std::int32_t> dim{};
  const legate::mapping::DimOrdering order{};
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(DimOrderingTest, CreateCOrder)
{
  const std::vector<std::int32_t> dim{};
  const auto c_order = legate::mapping::DimOrdering::c_order();
  check_dim_ordering(c_order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(DimOrderingTest, CreateFortranOrder)
{
  const std::vector<std::int32_t> dim{};
  const auto fortran_order = legate::mapping::DimOrdering::fortran_order();
  check_dim_ordering(fortran_order, legate::mapping::DimOrdering::Kind::FORTRAN, dim);
}

TEST_F(DimOrderingTest, CreateCustomOrder)
{
  const std::vector<std::int32_t> dim_custom{0, 1, 2};
  const auto custom_order = legate::mapping::DimOrdering::custom_order(dim_custom);
  check_dim_ordering(custom_order, legate::mapping::DimOrdering::Kind::CUSTOM, dim_custom);
}

TEST_F(DimOrderingTest, GenerateLegionDimsCOrder)
{
  const std::vector<Legion::DimensionKind> expected{
    LEGION_DIM_Z, LEGION_DIM_Y, LEGION_DIM_X, LEGION_DIM_F};

  check_legion_dims(legate::mapping::DimOrdering::c_order(), /*ndim=*/3, expected);
}

TEST_F(DimOrderingTest, GenerateLegionDimsFortranOrder)
{
  const std::vector<Legion::DimensionKind> expected{
    LEGION_DIM_X, LEGION_DIM_Y, LEGION_DIM_Z, LEGION_DIM_F};

  check_legion_dims(legate::mapping::DimOrdering::fortran_order(), /*ndim=*/3, expected);
}

TEST_F(DimOrderingTest, GenerateLegionDimsCustomOrder)
{
  const std::vector<std::int32_t> dims{1, 2, 0};
  const std::vector<Legion::DimensionKind> expected{
    LEGION_DIM_Y, LEGION_DIM_Z, LEGION_DIM_X, LEGION_DIM_F};

  check_legion_dims(legate::mapping::DimOrdering::custom_order(dims), /*ndim=*/3, expected);
}

TEST_F(DimOrderingTest, GenerateLegionDimsForTransformedEmptyStore)
{
  MockMapperRuntime runtime;
  const auto context = Legion::Mapping::MapperContext{};
  const auto type    = legate::InternalSharedPtr<legate::detail::Type>{legate::int32().impl()};
  constexpr std::int32_t store_dim        = 0;
  constexpr std::int32_t region_field_dim = 1;
  constexpr Legion::FieldID field_id      = 1;
  const Legion::RegionRequirement region_requirement{};
  const auto region_field = legate::mapping::detail::RegionField{
    region_requirement, region_field_dim, /*idx=*/0, field_id, /*unbound=*/false};
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Project>(/*dim=*/0, /*coord=*/0),
    legate::make_internal_shared<legate::detail::TransformStack>());
  const auto store = legate::mapping::detail::Store{runtime,
                                                    context,
                                                    store_dim,
                                                    type,
                                                    legate::GlobalRedopID{0},
                                                    region_field,
                                                    /*is_unbound_store=*/false,
                                                    std::move(transform)};
  const std::vector<Legion::DimensionKind> expected{LEGION_DIM_X, LEGION_DIM_F};

  ASSERT_EQ(legate::mapping::DimOrdering::c_order().impl()->generate_legion_dims(store), expected);
}

TEST_F(DimOrderingTest, SetCOrder)
{
  const std::vector<std::int32_t> dim{};
  legate::mapping::DimOrdering order{};

  order.set_c_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(DimOrderingTest, SetFortranOrder)
{
  const std::vector<std::int32_t> dim{};
  legate::mapping::DimOrdering order{};

  order.set_fortran_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::FORTRAN, dim);
}

TEST_F(DimOrderingTest, SetCustomOrder)
{
  const std::vector<std::int32_t> dim_custom{0, 1, 2};
  legate::mapping::DimOrdering order{};

  order.set_custom_order(dim_custom);
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::CUSTOM, dim_custom);
}

TEST_F(DimOrderingTest, SetOrderMultipleTimes)
{
  const std::vector<std::int32_t> dim{};
  const std::vector<std::int32_t> dim_custom{0, 1, 2};
  legate::mapping::DimOrdering order{};

  // Set to fortran, then custom, then back to c
  order.set_fortran_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::FORTRAN, dim);

  order.set_custom_order(dim_custom);
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::CUSTOM, dim_custom);

  order.set_c_order();
  check_dim_ordering(order, legate::mapping::DimOrdering::Kind::C, dim);
}

TEST_F(DimOrderingTest, CopyAssignment)
{
  const std::vector<std::int32_t> dims{1, 0, 2};
  const auto source = legate::mapping::DimOrdering::custom_order(dims);
  auto target       = legate::mapping::DimOrdering::c_order();

  target = source;

  check_dim_ordering(target, legate::mapping::DimOrdering::Kind::CUSTOM, dims);
  ASSERT_EQ(target, source);
}

TEST_F(DimOrderingTest, Equal)
{
  const legate::mapping::DimOrdering order1{};
  const legate::mapping::DimOrdering order2{};

  ASSERT_EQ(order1, order2);
}

TEST_F(DimOrderingTest, NotEqual)
{
  legate::mapping::DimOrdering order1{};
  legate::mapping::DimOrdering order2{};
  legate::mapping::DimOrdering order3{};

  order1.set_custom_order({});
  order2.set_fortran_order();
  order3.set_custom_order({0, 1});

  ASSERT_NE(order1, order2);
  ASSERT_NE(order1, order3);
  ASSERT_NE(order2, order3);
}

}  // namespace dim_ordering_unit
