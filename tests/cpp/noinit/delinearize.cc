/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <legate/utilities/detail/linearize.h>

#include <fmt/format.h>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <utilities/utilities.h>

namespace test_linearize_delinearize {

namespace {

template <typename DIM_TYPE>
class LinearizeUnit : public DefaultFixture {
  static_assert(DIM_TYPE::value > 0);
};

class DimTypes {
  // Note the 0, we skip it since we cannot have 0-dimensional points
  template <std::size_t... DIM>
  static constexpr auto detect_(std::index_sequence<0, DIM...>)
    -> ::testing::Types<std::integral_constant<std::int32_t, DIM>...>
  {
  }

 public:
  using type = decltype(detect_(std::make_index_sequence<LEGATE_MAX_DIM>{}));

  template <typename T>
  static std::string GetName(int)  // NOLINT(readability-identifier-naming)
  {
    return fmt::format("{}D", T::value);
  }
};

}  // namespace

TYPED_TEST_SUITE(LinearizeUnit, DimTypes::type, DimTypes);

TYPED_TEST(LinearizeUnit, Linearize)
{
  static constexpr auto DIM = TypeParam::value;
  const auto POINT_MIN      = legate::Point<DIM>::ZEROES();
  const auto POINT_MAX      = legate::Point<DIM>::ONES() + legate::Point<DIM>::ONES();
  const auto rect           = legate::Rect<DIM>{POINT_MIN, POINT_MAX};
  std::size_t idx           = 0;

  // Use row-major ordering for the traversal. This doesn't affect the output, but it makes
  // asserting the value of the linearized point easier. Otherwise, a 2x2 rect would be
  // traversed as:
  //
  // Point  -> idx
  // <0, 0> -> 0
  // <1, 0> -> 3
  // <2, 0> -> 6
  // <0, 1> -> 1
  // <1, 1> -> 4
  // <2, 1> -> 7
  // <0, 2> -> 2
  // <1, 2> -> 5
  // <2, 2> -> 8
  for (auto it = legate::PointInRectIterator<DIM>{rect, /* column_major_order */ false}; it.valid();
       ++it) {
    const auto lin_idx = legate::detail::linearize(rect.lo, rect.hi, *it);

    ASSERT_EQ(lin_idx, idx) << "lo = " << rect.lo << ", hi = " << rect.hi << ", point = " << *it;
    ++idx;
  }
}

TYPED_TEST(LinearizeUnit, Delinearize)
{
  static constexpr auto DIM = TypeParam::value;
  const auto POINT_MIN      = legate::Point<DIM>::ZEROES();
  const auto POINT_MAX      = legate::Point<DIM>::ONES() + legate::Point<DIM>::ONES();
  const auto rect           = legate::Rect<DIM>{POINT_MIN, POINT_MAX};
  auto it = legate::PointInRectIterator<DIM>{rect, /* column_major_order */ false};

  for (std::size_t idx = 0; idx < rect.volume(); ++idx, static_cast<void>(++it)) {
    const auto delin_pt = legate::detail::delinearize(rect.lo, rect.hi, idx);

    ASSERT_TRUE(it.valid());
    ASSERT_EQ(delin_pt, *it);
  }
}

TYPED_TEST(LinearizeUnit, RoundTrip)
{
  static constexpr auto DIM = TypeParam::value;
  const auto POINT_MIN      = legate::Point<DIM>::ZEROES();
  const auto POINT_MAX      = legate::Point<DIM>::ONES() + legate::Point<DIM>::ONES();
  const auto rect           = legate::Rect<DIM>{POINT_MIN, POINT_MAX};

  for (auto it = legate::PointInRectIterator<DIM>{rect, /* column_major_order */ false}; it.valid();
       ++it) {
    const auto lin_idx  = legate::detail::linearize(rect.lo, rect.hi, *it);
    const auto delin_pt = legate::detail::delinearize(rect.lo, rect.hi, lin_idx);

    ASSERT_EQ(delin_pt, *it);
  }
}

}  // namespace test_linearize_delinearize
