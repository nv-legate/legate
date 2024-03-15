/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/utilities/internal_shared_ptr.h"

#include <gtest/gtest.h>

using BasicSharedPtrTypeList = ::testing::Types<std::int8_t, std::int32_t, std::uint64_t>;

template <typename>
struct InternalWeakPtrUnit : ::testing::Test {};

TYPED_TEST_SUITE(InternalWeakPtrUnit, BasicSharedPtrTypeList, );

TYPED_TEST(InternalWeakPtrUnit, Create)
{
  legate::InternalWeakPtr<TypeParam> ptr;

  EXPECT_EQ(ptr.use_count(), 0);
  EXPECT_TRUE(ptr.expired());
  EXPECT_EQ(ptr.lock(), legate::InternalSharedPtr<TypeParam>{});
}

TYPED_TEST(InternalWeakPtrUnit, CreateFromSharedPtrCtor)
{
  auto shared_ptr = legate::make_internal_shared<TypeParam>(TypeParam{66});
  legate::InternalWeakPtr<TypeParam> ptr{shared_ptr};

  EXPECT_EQ(ptr.use_count(), 1);
  EXPECT_FALSE(ptr.expired());
  EXPECT_EQ(ptr.lock(), shared_ptr);
}

TYPED_TEST(InternalWeakPtrUnit, CreateFromSharedPtrEq)
{
  legate::InternalWeakPtr<TypeParam> ptr;
  auto shared_ptr = legate::make_internal_shared<TypeParam>(TypeParam{66});

  ptr = shared_ptr;

  EXPECT_EQ(ptr.use_count(), 1);
  EXPECT_FALSE(ptr.expired());
  EXPECT_EQ(ptr.lock(), shared_ptr);
}

TYPED_TEST(InternalWeakPtrUnit, CreateFromSharedPtrDrop)
{
  legate::InternalWeakPtr<TypeParam> ptr;

  {
    auto shared_ptr = legate::make_internal_shared<TypeParam>(TypeParam{66});

    ptr = shared_ptr;

    EXPECT_EQ(ptr.use_count(), 1);
    EXPECT_FALSE(ptr.expired());
    EXPECT_EQ(ptr.lock(), shared_ptr);
  }
  EXPECT_TRUE(ptr.expired());
  EXPECT_EQ(ptr.use_count(), 0);
  EXPECT_EQ(ptr.lock(), legate::InternalSharedPtr<TypeParam>{});
}

TYPED_TEST(InternalWeakPtrUnit, CreateFromEmptySharedPtrDrop)
{
  legate::InternalWeakPtr<TypeParam> ptr;

  {
    // This shared ptr owns nothing, so the weak ptr shouldn't either
    legate::InternalSharedPtr<TypeParam> shared_ptr;

    ptr = shared_ptr;

    EXPECT_EQ(ptr.use_count(), 0);
    EXPECT_TRUE(ptr.expired());
    EXPECT_EQ(ptr.lock(), shared_ptr);
  }
  EXPECT_TRUE(ptr.expired());
  EXPECT_EQ(ptr.use_count(), 0);
  EXPECT_EQ(ptr.lock(), legate::InternalSharedPtr<TypeParam>{});
}

TYPED_TEST(InternalWeakPtrUnit, Swap)
{
  auto shared_1 = legate::make_internal_shared<TypeParam>(TypeParam{66});
  auto shared_2 = legate::make_internal_shared<TypeParam>(TypeParam{88});

  legate::InternalWeakPtr<TypeParam> weak_1{shared_1};
  legate::InternalWeakPtr<TypeParam> weak_2{shared_2};

  const auto equal_ptrs = [](const legate::InternalWeakPtr<TypeParam>& weak,
                             const legate::InternalSharedPtr<TypeParam>& shared) {
    EXPECT_EQ(weak.use_count(), 1);
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(weak.lock(), shared);
    EXPECT_EQ(*(weak.lock()), *shared);
  };

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);

  weak_1.swap(weak_2);

  // weak_1 == shared_2, weak_2 == shared_1
  equal_ptrs(weak_1, shared_2);
  equal_ptrs(weak_2, shared_1);

  weak_1.swap(weak_2);

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);

  // double swap should do nothing
  weak_1.swap(weak_2);
  weak_1.swap(weak_2);

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);

  // doesn't matter who swaps
  weak_2.swap(weak_1);
  weak_2.swap(weak_1);

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);

  // self-swap is also a no-op
  weak_1.swap(weak_1);

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);

  using std::swap;

  swap(weak_1, weak_2);

  // weak_1 == shared_2, weak_2 == shared_1
  equal_ptrs(weak_1, shared_2);
  equal_ptrs(weak_2, shared_1);

  swap(weak_1, weak_2);

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);

  swap(weak_1, weak_2);
  swap(weak_1, weak_2);

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);

  swap(weak_2, weak_1);
  swap(weak_2, weak_1);

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);

  // self-swap OK
  swap(weak_1, weak_1);

  // weak_1 == shared_1, weak_2 == shared_2
  equal_ptrs(weak_1, shared_1);
  equal_ptrs(weak_2, shared_2);
}
