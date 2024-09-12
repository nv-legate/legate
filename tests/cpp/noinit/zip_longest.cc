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

// NOLINTBEGIN(readability-magic-numbers)

class ZipEqualFn {
 public:
  template <typename... T>
  [[nodiscard]] auto operator()(T&&... args) const
  {
    return legate::detail::zip_equal(std::forward<T>(args)...);
  }
};

using ZipTester = zip_iterator_common::ZipTester<ZipEqualFn>;

TEST(ZipEqualUnit, Construct) { ZipTester::construct_test(); }

TEST(ZipEqualUnit, IterateEmpty) { ZipTester::empty_test(); }

TEST(ZipEqualUnit, IterateAllSameSize) { ZipTester::same_size_test(); }

TEST(ZipEqualUnit, IterateAllSameSizeModify) { ZipTester::all_same_size_modify_test(); }

TEST(ZipEqualUnit, RandomAccess) { ZipTester::random_access_test(); }

TEST(ZipEqualUnit, Relational) { ZipTester::relational_test(); }

// NOLINTEND(readability-magic-numbers)

}  // namespace zip_longest_test
