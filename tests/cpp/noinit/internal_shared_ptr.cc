/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Must go first
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#define LEGATE_INTERNAL_SHARED_PTR_TESTS 1
//
#include <legate/utilities/internal_shared_ptr.h>

#include <cstdint>
#include <cstring>
#include <noinit/shared_ptr_util.h>
#include <stdexcept>
#include <thread>

template <typename T>
struct InternalSharedPtrUnit : BasicSharedPtrUnit<T> {};

// NOLINTBEGIN(readability-magic-numbers)

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

TYPED_TEST(InternalSharedPtrUnit, CreateWithWeakPtrNegative)
{
  legate::InternalWeakPtr<TypeParam> weak_ptr{};

  ASSERT_THAT(
    [&] {
      const legate::InternalSharedPtr<TypeParam> ptr{weak_ptr};
      static_cast<void>(ptr);
    },
    ::testing::ThrowsMessage<legate::BadInternalWeakPtr>(::testing::HasSubstr(
      "Trying to construct an InternalSharedPtr from an empty InternalWeakPtr")));
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

  const legate::InternalSharedPtr<TypeParam> sh_ptr{std::move(uniq)};

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

  std::fill(bare_ptr, bare_ptr + N, 1);

  legate::InternalSharedPtr<TypeParam[]> ptr{bare_ptr};

  test_basic_equal(ptr, bare_ptr, N);
}

namespace legate {

class InternalSharedPtrUnitFriend : public BasicSharedPtrUnit<> {};

namespace {

constexpr const char EXCEPTION_TEXT[] = "There is no peace but the Pax Romana";

template <typename T>
class ThrowingAllocator {
 public:
  using size_type  = std::size_t;
  using value_type = T;

  constexpr ThrowingAllocator() noexcept = default;

  template <typename U>
  constexpr ThrowingAllocator(ThrowingAllocator<U>) noexcept  // NOLINT(google-explicit-constructor)
  {
  }

  [[nodiscard]] static T* allocate(size_type, const void* = nullptr)
  {
    throw std::runtime_error{EXCEPTION_TEXT};
  }

  static void deallocate(const void* ptr, size_type n = 1)
  {
    FAIL() << "Trying to deallocate " << ptr << " (size " << n << ") from ThrowingAllocator";
  }
};

class DeleterChecker {
 public:
  explicit DeleterChecker(bool* target) : deleted_{target} {}

  void operator()(void* /*ptr*/) const { *deleted_ = true; }

 private:
  bool* deleted_{};
};

}  // namespace

TEST_F(InternalSharedPtrUnitFriend, UniqThrow)
{
  constexpr int val = 123;
  auto uniq         = std::make_unique<int>(val);
  auto ptr          = uniq.get();
  auto deleter      = uniq.get_deleter();

  ASSERT_NE(uniq.get(), nullptr);
  ASSERT_TRUE(uniq);
  ASSERT_EQ(*uniq, val);

  ASSERT_THAT(
    ([&] {
      const InternalSharedPtr<int> sh_ptr{
        InternalSharedPtr<int>::NoCatchAndDeleteTag{}, ptr, deleter, ThrowingAllocator<int>{}};
      static_cast<void>(sh_ptr);
    }),
    ::testing::ThrowsMessage<std::runtime_error>(::testing::StrEq(EXCEPTION_TEXT)));

  ASSERT_NE(uniq.get(), nullptr);
  ASSERT_TRUE(uniq);
  ASSERT_EQ(*uniq, val);
}

TEST_F(InternalSharedPtrUnitFriend, UniqThrowDeleted)
{
  constexpr int val = 123;
  auto uniq         = std::make_unique<int>(val);
  auto ptr          = uniq.get();
  auto deleted      = false;
  auto deleter      = DeleterChecker{&deleted};

  ASSERT_FALSE(deleted);

  ASSERT_THAT(([&] {
                const InternalSharedPtr<int> sh_ptr{ptr, deleter, ThrowingAllocator<int>{}};
                static_cast<void>(sh_ptr);
              }),
              ::testing::ThrowsMessage<std::runtime_error>(::testing::StrEq(EXCEPTION_TEXT)));

  ASSERT_TRUE(deleted);
}

}  // namespace legate

TYPED_TEST(InternalSharedPtrUnit, MultiThreaded)
{
  auto sh_ptr                  = legate::make_internal_shared<TypeParam>(123);
  std::atomic<bool> t1_started = false;
  std::exception_ptr t1exn     = nullptr;
  std::exception_ptr t2exn     = nullptr;
  constexpr auto max_it        = 1'000;

  auto t1 = std::thread{[&, other_ptr = sh_ptr, sh_ptr]() mutable {
    t1_started = true;
    try {
      for (auto i = 0; i < max_it; ++i) {
        other_ptr = sh_ptr;
        ASSERT_TRUE(sh_ptr.use_count());
        ASSERT_TRUE(sh_ptr.get());
        ASSERT_EQ(sh_ptr.get(), other_ptr.get());
        ASSERT_EQ(*sh_ptr, *other_ptr);
        sh_ptr = legate::SharedPtr<TypeParam>{other_ptr};
        other_ptr.reset();
      }
    } catch (...) {
      t1exn = std::current_exception();
    }
  }};

  auto t2 = std::thread{[&, sh_ptr_2 = std::move(sh_ptr)]() mutable {
    // Wait for other thread to definitely have started so we at least make it likely they both
    // run at the same time
    while (!t1_started) {}
    try {
      auto other_ptr = sh_ptr_2;

      for (auto i = 0; i < max_it; ++i) {
        other_ptr = sh_ptr_2;
        sh_ptr_2  = other_ptr;
        ASSERT_TRUE(sh_ptr_2.use_count());
        ASSERT_TRUE(sh_ptr_2.get());
        ASSERT_EQ(sh_ptr_2.get(), other_ptr.get());
        ASSERT_EQ(*sh_ptr_2, *other_ptr);
        other_ptr.reset();
      }
    } catch (...) {
      t2exn = std::current_exception();
    }
  }};

  t2.join();
  t1.join();

  constexpr auto check_exn = [](const std::exception_ptr& ptr, int thread_num) {
    try {
      if (ptr) {
        std::rethrow_exception(ptr);
      }
    } catch (const std::exception& e) {
      FAIL() << "Exception thrown in thread " << thread_num << ": " << e.what() << "\n";
    }
  };
  check_exn(t1exn, 1);
  check_exn(t2exn, 2);
}

TEST(InternalSharedPtrUnit, DynamicPointerCastConstRef)
{
  enum class Kind : std::uint8_t { BASE, DERIVED };

  class BaseFoo {
   public:
    [[nodiscard]] virtual Kind kind() const { return Kind::BASE; }

    virtual ~BaseFoo() = default;
  };

  class DerivedFoo : public BaseFoo {
   public:
    [[nodiscard]] Kind kind() const override { return Kind::DERIVED; }
  };

  // Use new expression so we can construct a base shared ptr directly.
  const auto der = legate::InternalSharedPtr<BaseFoo>{new DerivedFoo};

  ASSERT_NE(der, nullptr);
  ASSERT_EQ(der->kind(), Kind::DERIVED);

  const auto dyn_cast = legate::dynamic_pointer_cast<DerivedFoo>(der);

  ASSERT_NE(dyn_cast, nullptr);
  ASSERT_EQ(dyn_cast->kind(), Kind::DERIVED);
  // Dyn-casting should not modify the original
  ASSERT_NE(der, nullptr);
  ASSERT_EQ(der->kind(), Kind::DERIVED);

  class Unrelated {};

  const auto unrelated = legate::dynamic_pointer_cast<Unrelated>(der);

  // Unrelated, should fail
  ASSERT_EQ(unrelated, nullptr);
  // Dyn-casting should not modify the original
  ASSERT_NE(der, nullptr);
  ASSERT_EQ(der->kind(), Kind::DERIVED);
}

TEST(InternalSharedPtrUnit, DynamicPointerCastMove)
{
  enum class Kind : std::uint8_t { BASE, DERIVED };

  class BaseFoo {
   public:
    [[nodiscard]] virtual Kind kind() const { return Kind::BASE; }

    virtual ~BaseFoo() = default;
  };

  class DerivedFoo : public BaseFoo {
   public:
    [[nodiscard]] Kind kind() const override { return Kind::DERIVED; }
  };

  // Use new expression so we can construct a base shared ptr directly.
  auto der = legate::InternalSharedPtr<BaseFoo>{new DerivedFoo};

  ASSERT_NE(der, nullptr);
  ASSERT_EQ(der->kind(), Kind::DERIVED);

  auto dyn_cast = legate::dynamic_pointer_cast<DerivedFoo>(std::move(der));

  ASSERT_NE(dyn_cast, nullptr);
  ASSERT_EQ(dyn_cast->kind(), Kind::DERIVED);
  // We *successfully* moved, so der should be NULL
  ASSERT_EQ(der, nullptr);  // NOLINT(bugprone-use-after-move)

  class Unrelated {};

  const auto unrelated = legate::dynamic_pointer_cast<Unrelated>(std::move(dyn_cast));

  // Unrelated, should fail
  ASSERT_EQ(unrelated, nullptr);
  // We failed the case moved, so dyn_cast should be non-NULL
  ASSERT_NE(dyn_cast, nullptr);                // NOLINT(bugprone-use-after-move)
  ASSERT_EQ(dyn_cast->kind(), Kind::DERIVED);  // NOLINT(bugprone-use-after-move)
}

TEST(InternalSharedPtrUnit, ReinterpretPointerCastConstRef)
{
  static constexpr std::int32_t MINUS_ONE = -1;
  const std::uint32_t UINT_MINUS_ONE      = [] {
    std::uint32_t ret{};

    // Need to bitcast as that is what the reinterpret_pointer_cast() will equate to.
    static_assert(sizeof(ret) == sizeof(MINUS_ONE));
    std::memcpy(&ret, &MINUS_ONE, sizeof(ret));
    return ret;
  }();

  const auto int_ptr = legate::make_internal_shared<std::int32_t>(MINUS_ONE);

  ASSERT_NE(int_ptr, nullptr);
  ASSERT_EQ(*int_ptr, MINUS_ONE);

  const auto uint_ptr = legate::reinterpret_pointer_cast<std::uint32_t>(int_ptr);

  ASSERT_NE(uint_ptr, nullptr);
  ASSERT_EQ(*uint_ptr, UINT_MINUS_ONE);
  // Reinterpret casting should have no effect on the original
  ASSERT_NE(int_ptr, nullptr);
  ASSERT_EQ(*int_ptr, MINUS_ONE);
}

TEST(InternalSharedPtrUnit, ReinterpretPointerCastMove)
{
  static constexpr std::int32_t MINUS_ONE = -1;
  const std::uint32_t UINT_MINUS_ONE      = [] {
    std::uint32_t ret{};

    // Need to bitcast as that is what the reinterpret_pointer_cast() will equate to.
    static_assert(sizeof(ret) == sizeof(MINUS_ONE));
    std::memcpy(&ret, &MINUS_ONE, sizeof(ret));
    return ret;
  }();

  auto int_ptr = legate::make_internal_shared<std::int32_t>(MINUS_ONE);

  ASSERT_NE(int_ptr, nullptr);
  ASSERT_EQ(*int_ptr, MINUS_ONE);

  const auto uint_ptr = legate::reinterpret_pointer_cast<std::uint32_t>(std::move(int_ptr));

  ASSERT_NE(uint_ptr, nullptr);
  ASSERT_EQ(*uint_ptr, UINT_MINUS_ONE);
  // We moved, so should be NULL
  ASSERT_EQ(int_ptr, nullptr);  // NOLINT(bugprone-use-after-move)
}

// NOLINTEND(readability-magic-numbers)
