/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/transform/transform_stack.h>
#include <legate/data/detail/transform/transpose.h>
#include <legate/mapping/detail/store.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace mapping_store_test {

using legate::mapping::detail::FutureWrapper;
using legate::mapping::detail::Store;

namespace {

using MappingStoreTransformTest = DefaultFixture;

}  // namespace

TEST_F(MappingStoreTransformTest, FutureWithTransform)
{
  constexpr std::int32_t dim = 2;
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::float64().impl()};
  const Legion::Domain domain{Legion::Rect<2>{{0, 0}, {7, 7}}};
  const FutureWrapper future{/*idx=*/1, domain};
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();
  const Store store{dim, type, future, std::move(transform)};

  ASSERT_TRUE(store.is_future());
  ASSERT_EQ(store.dim(), dim);
  ASSERT_TRUE(store.valid());
}

TEST_F(MappingStoreTransformTest, FindImaginaryDimsNoTransform)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};
  const Store store{/*dim=*/1, type, future};
  auto imaginary_dims = store.find_imaginary_dims();

  ASSERT_TRUE(imaginary_dims.empty());
}

TEST_F(MappingStoreTransformTest, FindImaginaryDimsWithIdentityTransform)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();
  const Store store{/*dim=*/1, type, future, std::move(transform)};
  auto imaginary_dims = store.find_imaginary_dims();

  // Identity transform should have no imaginary dims
  ASSERT_TRUE(imaginary_dims.empty());
}

TEST_F(MappingStoreTransformTest, InvertDimsNoTransform)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<2>{{0, 0}, {9, 9}}};
  const FutureWrapper future{/*idx=*/0, domain};
  const Store store{/*dim=*/2, type, future};
  const legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> dims{0, 1};
  auto inverted = store.invert_dims(dims);

  ASSERT_THAT(inverted, ::testing::ElementsAre(0, 1));
}

TEST_F(MappingStoreTransformTest, InvertDimsWithIdentityTransform)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<2>{{0, 0}, {9, 9}}};
  const FutureWrapper future{/*idx=*/0, domain};
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>();
  const Store store{/*dim=*/2, type, future, std::move(transform)};
  const legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> dims{1, 0};
  auto inverted = store.invert_dims(dims);

  // Identity transform should return the same dims
  ASSERT_THAT(inverted, ::testing::ElementsAre(1, 0));
}

TEST_F(MappingStoreTransformTest, InvertDimsWithTranspose)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<2>{{0, 0}, {9, 9}}};
  const FutureWrapper future{/*idx=*/0, domain};

  // Create a Transpose transform: axes {1, 0} means swap dimensions
  // Logical dim 0 -> Physical dim 1
  // Logical dim 1 -> Physical dim 0
  auto parent              = legate::make_internal_shared<legate::detail::TransformStack>();
  auto transpose_transform = std::make_unique<legate::detail::Transpose>(
    legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM>{1, 0});
  auto transform = legate::make_internal_shared<legate::detail::TransformStack>(
    std::move(transpose_transform), parent);

  const Store store{/*dim=*/2, type, future, std::move(transform)};

  // Test with C-order dims {0, 1}
  const legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> dims_c_order{0, 1};
  auto inverted_c = store.invert_dims(dims_c_order);
  // After transpose inversion: {1, 0} (becomes Fortran order)
  ASSERT_THAT(inverted_c, ::testing::ElementsAre(1, 0));

  // Test with Fortran-order dims {1, 0}
  const legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> dims_f_order{1, 0};
  auto inverted_f = store.invert_dims(dims_f_order);
  // After transpose inversion: {0, 1} (becomes C order)
  ASSERT_THAT(inverted_f, ::testing::ElementsAre(0, 1));
}

TEST_F(MappingStoreTransformTest, InvertDimsEmpty)
{
  const legate::InternalSharedPtr<legate::detail::Type> type{legate::int32().impl()};
  const Legion::Domain domain{Legion::Rect<1>{0, 9}};
  const FutureWrapper future{/*idx=*/0, domain};
  const Store store{/*dim=*/1, type, future};
  const legate::detail::SmallVector<std::int32_t, LEGATE_MAX_DIM> dims{};
  auto inverted = store.invert_dims(dims);

  ASSERT_TRUE(inverted.empty());
}

}  // namespace mapping_store_test
