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

#include "shared_ptr_util.h"

template <typename T>
struct InternalSharedPtrUnit : BasicSharedPtrUnit<T> {};

TYPED_TEST_SUITE(InternalSharedPtrUnit, BasicSharedPtrTypeList, );

TYPED_TEST(InternalSharedPtrUnit, CreateBasic)
{
  legate::InternalSharedPtr<TypeParam> ptr;

  test_basic_equal(ptr, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(InternalSharedPtrUnit, CreateNullptrT)
{
  legate::InternalSharedPtr<TypeParam> ptr{nullptr};

  test_basic_equal(ptr, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(InternalSharedPtrUnit, CreateWithPtr)
{
  auto sh_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr{sh_ptr};

  EXPECT_EQ(ptr.use_count(), 1);
  test_basic_equal(ptr, sh_ptr);
}

TYPED_TEST(InternalSharedPtrUnit, CreateWithCopyEqCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};
  legate::InternalSharedPtr<TypeParam> ptr2 = ptr1;

  test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
}

// same test as above, but using {} constructor
TYPED_TEST(InternalSharedPtrUnit, CreateWithCopyBraceCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};
  legate::InternalSharedPtr<TypeParam> ptr2{ptr1};

  test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
}

TYPED_TEST(InternalSharedPtrUnit, CascadingCopyEqCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};
  {
    legate::InternalSharedPtr<TypeParam> ptr2 = ptr1;

    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
    {
      legate::InternalSharedPtr<TypeParam> ptr3 = ptr2;

      test_create_with_copy_n({ptr1, ptr2, ptr3}, bare_ptr);
    }
    // ensure that ref counts have decreased again
    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
    {
      // note initializing with ptr1 now
      legate::InternalSharedPtr<TypeParam> ptr3 = ptr1;

      test_create_with_copy_n({ptr1, ptr2, ptr3}, bare_ptr);
    }
    // ensure that ref counts have decreased again
    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
  }
  // ensure that ref counts have decreased again
  test_create_with_copy_n({ptr1}, bare_ptr);
}

TYPED_TEST(InternalSharedPtrUnit, CascadingCopyBraceCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};
  {
    legate::InternalSharedPtr<TypeParam> ptr2{ptr1};

    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
    {
      legate::InternalSharedPtr<TypeParam> ptr3{ptr2};

      test_create_with_copy_n({ptr1, ptr2, ptr3}, bare_ptr);
    }
    // ensure that ref counts have decreased again
    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
    {
      // note initializing with ptr1 now
      legate::InternalSharedPtr<TypeParam> ptr3{ptr1};

      test_create_with_copy_n({ptr1, ptr2, ptr3}, bare_ptr);
    }
    // ensure that ref counts have decreased again
    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
  }
  // ensure that ref counts have decreased again
  test_create_with_copy_n({ptr1}, bare_ptr);
}

TYPED_TEST(InternalSharedPtrUnit, MoveCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};

  test_basic_equal(ptr1, bare_ptr);

  legate::InternalSharedPtr<TypeParam> ptr2 = std::move(ptr1);

  EXPECT_EQ(ptr2.use_count(), 1);
  test_basic_equal(ptr2, bare_ptr);
  test_basic_equal(ptr1, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(InternalSharedPtrUnit, MoveAssign)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};

  test_basic_equal(ptr1, bare_ptr);

  legate::InternalSharedPtr<TypeParam> ptr2{std::move(ptr1)};

  EXPECT_EQ(ptr2.use_count(), 1);
  test_basic_equal(ptr2, bare_ptr);
  test_basic_equal(ptr1, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(InternalSharedPtrUnit, SelfAssign)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};
  // Use this silence compiler warnings about self-assignment, as that is indeed the point of
  // this test.
  auto hide_self_assign = [](auto& lhs, auto& rhs) { lhs = rhs; };

  hide_self_assign(ptr1, ptr1);
  EXPECT_EQ(ptr1.use_count(), 1);
  test_basic_equal(ptr1, bare_ptr);
}

TYPED_TEST(InternalSharedPtrUnit, SelfMoveAssign)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};
  // Use this silence compiler warnings about self-assignment, as that is indeed the point of
  // this test.
  auto hide_self_assign = [](auto& lhs, auto& rhs) { lhs = std::move(rhs); };

  hide_self_assign(ptr1, ptr1);
  EXPECT_EQ(ptr1.use_count(), 1);
  test_basic_equal(ptr1, bare_ptr);
}

TYPED_TEST(InternalSharedPtrUnit, Reset)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};

  test_basic_equal(ptr1, bare_ptr);
  ptr1.reset();
  test_basic_equal(ptr1, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(InternalSharedPtrUnit, ResetNullPtrT)
{
  auto bare_ptr = new TypeParam{11};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr};

  test_basic_equal(ptr1, bare_ptr);
  ptr1.reset(nullptr);
  test_basic_equal(ptr1, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(InternalSharedPtrUnit, ResetOther)
{
  auto bare_ptr1 = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> ptr1{bare_ptr1};

  test_basic_equal(ptr1, bare_ptr1);
  auto bare_ptr2 = new TypeParam{88};
  ptr1.reset(bare_ptr2);
  test_basic_equal(ptr1, bare_ptr2);
}

TEST(InternalSharedPtrUnit, BasicPolymorphism)
{
  auto bare_ptr = new BasicDerived{};
  legate::InternalSharedPtr<Base> ptr{bare_ptr};

  test_basic_equal(ptr, bare_ptr);
}

TEST(InternalSharedPtrUnit, Polymorphism)
{
  bool toggle = false;
  {
    auto bare_ptr = new TogglingDerived{&toggle};
    legate::InternalSharedPtr<Base> ptr{bare_ptr};

    ASSERT_FALSE(toggle);  // sanity check
    test_basic_equal(ptr, bare_ptr);
    ASSERT_FALSE(toggle);  // still false
    ASSERT_EQ(ptr.use_count(), 1);
  }
  ASSERT_TRUE(toggle);  // if properly handled, set to true in most derived dtor
}

TEST(InternalSharedPtrUnit, PolymorphismReset)
{
  bool toggle = false;
  {
    auto bare_ptr = new TogglingDerived{&toggle};
    legate::InternalSharedPtr<Base> ptr{bare_ptr};

    ASSERT_FALSE(toggle);  // sanity check
    test_basic_equal(ptr, bare_ptr);
    ASSERT_FALSE(toggle);  // still false
    ASSERT_EQ(ptr.use_count(), 1);

    auto bare_ptr2 = new BasicDerived{45};

    ptr.reset(bare_ptr2);
    ASSERT_TRUE(toggle);  // if properly handled, set to true in most derived dtor
    test_basic_equal(ptr, bare_ptr2);
    toggle = false;
  }
  ASSERT_FALSE(toggle);  // should not have been touched
}

TYPED_TEST(InternalSharedPtrUnit, MakeShared)
{
  auto sh_ptr   = legate::make_internal_shared<TypeParam>(10);
  auto bare_ptr = sh_ptr.get();

  test_basic_equal(sh_ptr, bare_ptr);
}

TEST(InternalSharedPtrUnit, MakeSharedPolymorphism)
{
  legate::InternalSharedPtr<Base> sh_ptr = legate::make_internal_shared<BasicDerived>(10);
  auto bare_ptr                          = static_cast<BasicDerived*>(sh_ptr.get());

  test_basic_equal(sh_ptr, bare_ptr);
}

TYPED_TEST(InternalSharedPtrUnit, UniqueCtor)
{
  auto val  = TypeParam{123};
  auto uniq = std::make_unique<TypeParam>(val);

  legate::InternalSharedPtr<TypeParam> sh_ptr{std::move(uniq)};

  ASSERT_EQ(sh_ptr.use_count(), 1);
  ASSERT_EQ(*sh_ptr, val);
  ASSERT_EQ(uniq.get(), nullptr);
  ASSERT_FALSE(uniq);
}

TYPED_TEST(InternalSharedPtrUnit, UniqueAssign)
{
  auto val  = TypeParam{123};
  auto uniq = std::make_unique<TypeParam>(val);

  auto bare_ptr = new TypeParam{22};
  legate::InternalSharedPtr<TypeParam> sh_ptr{bare_ptr};

  test_basic_equal(sh_ptr, bare_ptr);

  sh_ptr = std::move(uniq);

  ASSERT_EQ(sh_ptr.use_count(), 1);
  ASSERT_EQ(*sh_ptr, val);
  ASSERT_EQ(uniq.get(), nullptr);
  ASSERT_FALSE(uniq);
}

TYPED_TEST(InternalSharedPtrUnit, FromSharedCtor)
{
  auto sh_ptr = legate::make_shared<TypeParam>(88);

  ASSERT_EQ(sh_ptr.use_count(), 1);

  legate::InternalSharedPtr<TypeParam> internal_sh_ptr{sh_ptr};

  test_basic_equal(internal_sh_ptr, sh_ptr.get());

  ASSERT_EQ(internal_sh_ptr.as_user_ptr(), sh_ptr);
  ASSERT_EQ(internal_sh_ptr.use_count(), 2);
  ASSERT_EQ(internal_sh_ptr.strong_ref_count(), 2);
  ASSERT_EQ(internal_sh_ptr.user_ref_count(), 1);
  ASSERT_EQ(internal_sh_ptr.weak_ref_count(), 0);
}

TYPED_TEST(InternalSharedPtrUnit, FromSharedCtorMove)
{
  auto sh_ptr  = legate::make_shared<TypeParam>(88);
  auto raw_ptr = sh_ptr.get();

  ASSERT_EQ(sh_ptr.use_count(), 1);

  legate::InternalSharedPtr<TypeParam> internal_sh_ptr{std::move(sh_ptr)};

  test_basic_equal(internal_sh_ptr, raw_ptr);

  ASSERT_EQ(internal_sh_ptr.use_count(), 1);
  ASSERT_EQ(internal_sh_ptr.strong_ref_count(), 1);
  ASSERT_EQ(internal_sh_ptr.user_ref_count(), 0);
  ASSERT_EQ(internal_sh_ptr.weak_ref_count(), 0);
}

TYPED_TEST(InternalSharedPtrUnit, FromSharedAssign)
{
  auto sh_ptr = legate::make_shared<TypeParam>(88);

  ASSERT_EQ(sh_ptr.use_count(), 1);

  legate::InternalSharedPtr<TypeParam> internal_sh_ptr;

  test_basic_equal(internal_sh_ptr, static_cast<TypeParam*>(nullptr));
  internal_sh_ptr = sh_ptr;

  test_basic_equal(internal_sh_ptr, sh_ptr.get());

  ASSERT_EQ(internal_sh_ptr.as_user_ptr(), sh_ptr);
  ASSERT_EQ(internal_sh_ptr.use_count(), 2);
  ASSERT_EQ(internal_sh_ptr.strong_ref_count(), 2);
  ASSERT_EQ(internal_sh_ptr.user_ref_count(), 1);
  ASSERT_EQ(internal_sh_ptr.weak_ref_count(), 0);
}

TYPED_TEST(InternalSharedPtrUnit, FromSharedCtorMoveAssign)
{
  auto sh_ptr = legate::make_shared<TypeParam>(88);

  ASSERT_EQ(sh_ptr.use_count(), 1);

  legate::InternalSharedPtr<TypeParam> internal_sh_ptr;

  test_basic_equal(internal_sh_ptr, static_cast<TypeParam*>(nullptr));
  auto raw_ptr    = sh_ptr.get();
  internal_sh_ptr = std::move(sh_ptr);

  test_basic_equal(internal_sh_ptr, raw_ptr);

  ASSERT_EQ(internal_sh_ptr.use_count(), 1);
  ASSERT_EQ(internal_sh_ptr.strong_ref_count(), 1);
  ASSERT_EQ(internal_sh_ptr.user_ref_count(), 0);
  ASSERT_EQ(internal_sh_ptr.weak_ref_count(), 0);
}

TYPED_TEST(InternalSharedPtrUnit, Array)
{
  constexpr auto N = 100;
  auto bare_ptr    = new TypeParam[N];

  for (auto i = 0; i < N; ++i) {
    bare_ptr[i] = 1;
  }
  legate::InternalSharedPtr<TypeParam[]> ptr{bare_ptr};

  test_basic_equal(ptr, bare_ptr, N);
}
