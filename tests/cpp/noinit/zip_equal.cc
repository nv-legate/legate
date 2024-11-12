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

#include "legate/utilities/detail/zip.h"

#include "noinit/zip_common.h"

#include <gtest/gtest.h>
#include <utility>

namespace zip_longest_test {

namespace {

class ZipEqualFn {
 public:
  template <typename... T>
  [[nodiscard]] auto operator()(T&&... args) const
  {
    return legate::detail::zip_equal(std::forward<T>(args)...);
  }
};

using ZipTester = zip_iterator_common::ZipTester<ZipEqualFn>;

}  // namespace

namespace has_size_test {

static_assert(legate::traits::detail::is_detected_v<legate::detail::zip_detail::has_size,
                                                    std::vector<std::int32_t>>);
static_assert(
  !legate::traits::detail::is_detected_v<legate::detail::zip_detail::has_size, std::int32_t>);

static_assert(
  std::conjunction_v<legate::traits::detail::is_detected<legate::detail::zip_detail::has_size,
                                                         std::vector<std::int32_t>>,
                     legate::traits::detail::is_detected<legate::detail::zip_detail::has_size,
                                                         std::vector<std::int32_t>>>);

}  // namespace has_size_test

TEST(ZipEqual, BadSize)
{
  const std::vector<std::int32_t> v1{1, 2, 3, 4};
  const std::vector<std::int32_t> v2{1, 2, 3};

  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    // Throwing check is only performed in debug mode
    ASSERT_THROW(static_cast<void>(legate::detail::zip_equal(v1, v2)), std::invalid_argument);
  } else {
    ASSERT_NO_THROW(static_cast<void>(legate::detail::zip_equal(v1, v2)));
  }
}

TEST(ZipEqualUnit, Construct) { ZipTester::construct_test(); }

TEST(ZipEqualUnit, IterateEmpty) { ZipTester::empty_test(); }

TEST(ZipEqualUnit, IterateAllSameSize) { ZipTester::same_size_test(); }

TEST(ZipEqualUnit, IterateAllSameSizeModify) { ZipTester::all_same_size_modify_test(); }

TEST(ZipEqualUnit, RandomAccess) { ZipTester::random_access_test(); }

TEST(ZipEqualUnit, Relational) { ZipTester::relational_test(); }

}  // namespace zip_longest_test
