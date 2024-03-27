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

#include "core/utilities/shared_ptr.h"

#include "shared_ptr_util.h"

template <typename T>
struct SharedPtrUnit : BasicSharedPtrUnit<T> {};

TYPED_TEST_SUITE(SharedPtrUnit, BasicSharedPtrTypeList);

TYPED_TEST(SharedPtrUnit, CreateBasic)
{
  legate::SharedPtr<TypeParam> ptr;

  test_basic_equal(ptr, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(SharedPtrUnit, CreateNullptrT)
{
  legate::SharedPtr<TypeParam> ptr{nullptr};

  test_basic_equal(ptr, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(SharedPtrUnit, CreateWithPtr)
{
  auto sh_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr{sh_ptr};

  EXPECT_EQ(ptr.use_count(), 1);
  test_basic_equal(ptr, sh_ptr);
}

TYPED_TEST(SharedPtrUnit, CreateWithCopyEqCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};
  legate::SharedPtr<TypeParam> ptr2 = ptr1;

  test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
}

// same test as above, but using {} constructor
TYPED_TEST(SharedPtrUnit, CreateWithCopyBraceCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};
  legate::SharedPtr<TypeParam> ptr2{ptr1};

  test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, CascadingCopyEqCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};
  {
    legate::SharedPtr<TypeParam> ptr2 = ptr1;

    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
    {
      legate::SharedPtr<TypeParam> ptr3 = ptr2;

      test_create_with_copy_n({ptr1, ptr2, ptr3}, bare_ptr);
    }
    // ensure that ref counts have decreased again
    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
    {
      // note initializing with ptr1 now
      legate::SharedPtr<TypeParam> ptr3 = ptr1;

      test_create_with_copy_n({ptr1, ptr2, ptr3}, bare_ptr);
    }
    // ensure that ref counts have decreased again
    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
  }
  // ensure that ref counts have decreased again
  test_create_with_copy_n({ptr1}, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, CascadingCopyBraceCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};
  {
    legate::SharedPtr<TypeParam> ptr2{ptr1};

    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
    {
      legate::SharedPtr<TypeParam> ptr3{ptr2};

      test_create_with_copy_n({ptr1, ptr2, ptr3}, bare_ptr);
    }
    // ensure that ref counts have decreased again
    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
    {
      // note initializing with ptr1 now
      legate::SharedPtr<TypeParam> ptr3{ptr1};

      test_create_with_copy_n({ptr1, ptr2, ptr3}, bare_ptr);
    }
    // ensure that ref counts have decreased again
    test_create_with_copy_n({ptr1, ptr2}, bare_ptr);
  }
  // ensure that ref counts have decreased again
  test_create_with_copy_n({ptr1}, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, MoveCtor)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};

  test_basic_equal(ptr1, bare_ptr);

  legate::SharedPtr<TypeParam> ptr2 = std::move(ptr1);

  EXPECT_EQ(ptr2.use_count(), 1);
  test_basic_equal(ptr2, bare_ptr);
  test_basic_equal(ptr1, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(SharedPtrUnit, MoveAssign)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};

  test_basic_equal(ptr1, bare_ptr);

  legate::SharedPtr<TypeParam> ptr2{std::move(ptr1)};

  EXPECT_EQ(ptr2.use_count(), 1);
  test_basic_equal(ptr2, bare_ptr);
  test_basic_equal(ptr1, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(SharedPtrUnit, SelfAssign)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};
  // Use this silence compiler warnings about self-assignment, as that is indeed the point of
  // this test.
  auto hide_self_assign = [](auto& lhs, auto& rhs) { lhs = rhs; };

  hide_self_assign(ptr1, ptr1);
  EXPECT_EQ(ptr1.use_count(), 1);
  test_basic_equal(ptr1, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, SelfMoveAssign)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};
  // Use this silence compiler warnings about self-assignment, as that is indeed the point of
  // this test.
  auto hide_self_assign = [](auto& lhs, auto& rhs) { lhs = std::move(rhs); };

  hide_self_assign(ptr1, ptr1);
  EXPECT_EQ(ptr1.use_count(), 1);
  test_basic_equal(ptr1, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, Reset)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};

  test_basic_equal(ptr1, bare_ptr);
  ptr1.reset();
  test_basic_equal(ptr1, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(SharedPtrUnit, ResetNullPtrT)
{
  auto bare_ptr = new TypeParam{11};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr};

  test_basic_equal(ptr1, bare_ptr);
  ptr1.reset(nullptr);
  test_basic_equal(ptr1, static_cast<TypeParam*>(nullptr));
}

TYPED_TEST(SharedPtrUnit, ResetOther)
{
  auto bare_ptr1 = new TypeParam{1};
  legate::SharedPtr<TypeParam> ptr1{bare_ptr1};

  test_basic_equal(ptr1, bare_ptr1);
  auto bare_ptr2 = new TypeParam{88};
  ptr1.reset(bare_ptr2);
  test_basic_equal(ptr1, bare_ptr2);
}

TEST(SharedPtrUnit, BasicPolymorphism)
{
  auto bare_ptr = new BasicDerived{};
  legate::SharedPtr<Base> ptr{bare_ptr};

  test_basic_equal(ptr, bare_ptr);
}

TEST(SharedPtrUnit, Polymorphism)
{
  bool toggle = false;
  {
    auto bare_ptr = new TogglingDerived{&toggle};
    legate::SharedPtr<Base> ptr{bare_ptr};

    ASSERT_FALSE(toggle);  // sanity check
    test_basic_equal(ptr, bare_ptr);
    ASSERT_FALSE(toggle);  // still false
    ASSERT_EQ(ptr.use_count(), 1);
  }
  ASSERT_TRUE(toggle);  // if properly handled, set to true in most derived dtor
}

TYPED_TEST(SharedPtrUnit, CreateFromInternal)
{
  auto bare_ptr = new TypeParam{1};
  legate::InternalSharedPtr<TypeParam> itrnl_ptr{bare_ptr};
  legate::SharedPtr<TypeParam> sh_ptr{itrnl_ptr};

  ASSERT_EQ(itrnl_ptr.strong_ref_count(), 2);
  ASSERT_EQ(itrnl_ptr.user_ref_count(), 1);
  ASSERT_EQ(itrnl_ptr.use_count(), 2);
  ASSERT_EQ(sh_ptr.use_count(), itrnl_ptr.use_count());
  ASSERT_EQ(sh_ptr.get(), itrnl_ptr.get());
  ASSERT_EQ(*sh_ptr, *itrnl_ptr);
  ASSERT_EQ(static_cast<bool>(sh_ptr), static_cast<bool>(itrnl_ptr));
  test_basic_equal(itrnl_ptr, bare_ptr);
  test_basic_equal(sh_ptr, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, FromOrphanInternalCopy)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> sh_ptr;

  // sanity checks
  test_basic_equal(sh_ptr, static_cast<TypeParam*>(nullptr));
  {
    legate::InternalSharedPtr<TypeParam> itrnl_ptr{bare_ptr};

    sh_ptr = itrnl_ptr;
    ASSERT_EQ(itrnl_ptr.strong_ref_count(), 2);
    ASSERT_EQ(itrnl_ptr.user_ref_count(), 1);
    ASSERT_EQ(itrnl_ptr.use_count(), 2);
    ASSERT_EQ(sh_ptr.use_count(), 2);
    ASSERT_EQ(sh_ptr.use_count(), itrnl_ptr.use_count());
    ASSERT_EQ(sh_ptr.get(), itrnl_ptr.get());
    ASSERT_EQ(*sh_ptr, *itrnl_ptr);
    ASSERT_EQ(static_cast<bool>(sh_ptr), static_cast<bool>(itrnl_ptr));
    test_basic_equal(itrnl_ptr, bare_ptr);
    test_basic_equal(sh_ptr, bare_ptr);
  }
  // the pointer should not die here
  ASSERT_EQ(sh_ptr.use_count(), 1);
  test_basic_equal(sh_ptr, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, FromOrphanInternalCopyCascading)
{
  auto bare_ptr  = new TypeParam{1};
  auto bare_ptr2 = new TypeParam{2};
  legate::SharedPtr<TypeParam> sh_ptr;

  // sanity checks
  test_basic_equal(sh_ptr, static_cast<TypeParam*>(nullptr));
  {
    legate::InternalSharedPtr<TypeParam> itrnl_ptr{bare_ptr};

    sh_ptr = itrnl_ptr;
    ASSERT_EQ(itrnl_ptr.strong_ref_count(), 2);
    ASSERT_EQ(itrnl_ptr.user_ref_count(), 1);
    ASSERT_EQ(itrnl_ptr.use_count(), 2);
    ASSERT_EQ(sh_ptr.use_count(), 2);
    ASSERT_EQ(sh_ptr.use_count(), itrnl_ptr.use_count());
    ASSERT_EQ(sh_ptr.get(), itrnl_ptr.get());
    ASSERT_EQ(*sh_ptr, *itrnl_ptr);
    ASSERT_EQ(static_cast<bool>(sh_ptr), static_cast<bool>(itrnl_ptr));
    test_basic_equal(itrnl_ptr, bare_ptr);
    test_basic_equal(sh_ptr, bare_ptr);
    {
      legate::InternalSharedPtr<TypeParam> itrnl_ptr2{bare_ptr2};

      sh_ptr = itrnl_ptr2;
      ASSERT_EQ(itrnl_ptr2.strong_ref_count(), 2);
      ASSERT_EQ(itrnl_ptr2.user_ref_count(), 1);
      ASSERT_EQ(itrnl_ptr2.use_count(), 2);
      ASSERT_EQ(sh_ptr.use_count(), 2);
      ASSERT_EQ(sh_ptr.use_count(), itrnl_ptr2.use_count());
      ASSERT_EQ(sh_ptr.get(), itrnl_ptr2.get());
      ASSERT_EQ(*sh_ptr, *itrnl_ptr2);
      ASSERT_EQ(static_cast<bool>(sh_ptr), static_cast<bool>(itrnl_ptr2));
      test_basic_equal(itrnl_ptr2, bare_ptr2);
      test_basic_equal(sh_ptr, bare_ptr2);
    }
  }
  // the pointer should not die here
  ASSERT_EQ(sh_ptr.use_count(), 1);
  test_basic_equal(sh_ptr, bare_ptr2);
}

TYPED_TEST(SharedPtrUnit, FromOrphanInternalMove)
{
  auto bare_ptr = new TypeParam{1};
  legate::SharedPtr<TypeParam> sh_ptr;

  // sanity checks
  test_basic_equal(sh_ptr, static_cast<TypeParam*>(nullptr));
  {
    legate::InternalSharedPtr<TypeParam> itrnl_ptr{bare_ptr};

    sh_ptr = std::move(itrnl_ptr);
    // since itrnl_ptr has moved all of these should be ZERO
    ASSERT_EQ(itrnl_ptr.strong_ref_count(), 0);
    ASSERT_EQ(itrnl_ptr.user_ref_count(), 0);
    ASSERT_EQ(itrnl_ptr.use_count(), 0);
    ASSERT_EQ(sh_ptr.use_count(), 1);
    test_basic_equal(sh_ptr, bare_ptr);
    test_basic_equal(itrnl_ptr, static_cast<TypeParam*>(nullptr));
  }
  // the pointer should not die here
  ASSERT_EQ(sh_ptr.use_count(), 1);
  test_basic_equal(sh_ptr, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, FromOrphanInternalMoveCascading)
{
  auto bare_ptr  = new TypeParam{1};
  auto bare_ptr2 = new TypeParam{2};
  legate::SharedPtr<TypeParam> sh_ptr;

  // sanity checks
  test_basic_equal(sh_ptr, static_cast<TypeParam*>(nullptr));
  {
    legate::InternalSharedPtr<TypeParam> itrnl_ptr{bare_ptr};

    sh_ptr = std::move(itrnl_ptr);
    // since itrnl_ptr has moved all of these should be ZERO
    ASSERT_EQ(itrnl_ptr.strong_ref_count(), 0);
    ASSERT_EQ(itrnl_ptr.user_ref_count(), 0);
    ASSERT_EQ(itrnl_ptr.use_count(), 0);
    ASSERT_EQ(sh_ptr.use_count(), 1);
    test_basic_equal(sh_ptr, bare_ptr);
    test_basic_equal(itrnl_ptr, static_cast<TypeParam*>(nullptr));
    {
      legate::InternalSharedPtr<TypeParam> itrnl_ptr2{bare_ptr2};

      sh_ptr = std::move(itrnl_ptr2);
      // since itrnl_ptr has moved all of these should be ZERO
      ASSERT_EQ(itrnl_ptr2.strong_ref_count(), 0);
      ASSERT_EQ(itrnl_ptr2.user_ref_count(), 0);
      ASSERT_EQ(itrnl_ptr2.use_count(), 0);
      ASSERT_EQ(sh_ptr.use_count(), 1);
      test_basic_equal(sh_ptr, bare_ptr2);
      test_basic_equal(itrnl_ptr2, static_cast<TypeParam*>(nullptr));
    }
  }
  // the pointer should not die here
  ASSERT_EQ(sh_ptr.use_count(), 1);
  test_basic_equal(sh_ptr, bare_ptr2);
}

TYPED_TEST(SharedPtrUnit, InternalOutlives)
{
  auto bare_ptr = new TypeParam{77};
  legate::InternalSharedPtr<TypeParam> intrl_ptr{bare_ptr};

  test_basic_equal(intrl_ptr, bare_ptr);
  {
    legate::SharedPtr<TypeParam> sh_ptr{intrl_ptr};

    ASSERT_EQ(intrl_ptr.strong_ref_count(), 2);
    ASSERT_EQ(intrl_ptr.use_count(), 2);
    ASSERT_EQ(sh_ptr.use_count(), 2);
    ASSERT_EQ(intrl_ptr.user_ref_count(), 1);

    ASSERT_TRUE(intrl_ptr.get());
    ASSERT_EQ(intrl_ptr.get(), sh_ptr.get());
    ASSERT_EQ(*intrl_ptr, *sh_ptr);
    ASSERT_EQ(static_cast<bool>(intrl_ptr), static_cast<bool>(sh_ptr));
    test_basic_equal(sh_ptr, bare_ptr);
    test_basic_equal(intrl_ptr, bare_ptr);
  }
  ASSERT_EQ(intrl_ptr.strong_ref_count(), 1);
  ASSERT_EQ(intrl_ptr.user_ref_count(), 0);
  ASSERT_EQ(intrl_ptr.use_count(), 1);
  test_basic_equal(intrl_ptr, bare_ptr);
}

TYPED_TEST(SharedPtrUnit, MakeShared)
{
  auto sh_ptr   = legate::make_shared<TypeParam>(10);
  auto bare_ptr = sh_ptr.get();

  test_basic_equal(sh_ptr, bare_ptr);
}

TEST(SharedPtrUnit, MakeSharedPolymorphism)
{
  legate::SharedPtr<Base> sh_ptr = legate::make_shared<BasicDerived>(10);
  auto bare_ptr                  = static_cast<BasicDerived*>(sh_ptr.get());

  test_basic_equal(sh_ptr, bare_ptr);
}

TEST(SharedPtrUnit, PolymorphismReset)
{
  bool toggle = false;
  {
    auto bare_ptr = new TogglingDerived{&toggle};
    legate::SharedPtr<Base> ptr{bare_ptr};

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
  ASSERT_FALSE(toggle);  // toggle should not have been touched
}

TYPED_TEST(SharedPtrUnit, UniqueCtor)
{
  auto val  = TypeParam{123};
  auto uniq = std::make_unique<TypeParam>(val);

  legate::SharedPtr<TypeParam> sh_ptr{std::move(uniq)};

  ASSERT_EQ(sh_ptr.use_count(), 1);
  ASSERT_EQ(*sh_ptr, val);
  ASSERT_EQ(uniq.get(), nullptr);
  ASSERT_FALSE(uniq);
}

TYPED_TEST(SharedPtrUnit, UniqueAssign)
{
  auto val  = TypeParam{123};
  auto uniq = std::make_unique<TypeParam>(val);

  auto bare_ptr = new TypeParam{22};
  legate::SharedPtr<TypeParam> sh_ptr{bare_ptr};

  test_basic_equal(sh_ptr, bare_ptr);

  sh_ptr = std::move(uniq);

  ASSERT_EQ(sh_ptr.use_count(), 1);
  ASSERT_EQ(*sh_ptr, val);
  ASSERT_EQ(uniq.get(), nullptr);
  ASSERT_FALSE(uniq);
}
