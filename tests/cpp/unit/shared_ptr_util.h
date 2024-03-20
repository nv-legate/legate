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

#pragma once

#include "core/utilities/memory.h"

#include <array>
#include <cstdint>
#include <functional>
#include <gtest/gtest.h>
#include <vector>

struct UserType {
  std::array<char, 256> before_padding{};
  std::int32_t value{};
  std::array<char, 128> after_padding{};

  UserType() = default;

  explicit UserType(std::int32_t v) : value{v}
  {
    before_padding.fill(0);
    after_padding.fill(0);
  }

  UserType& operator=(std::int32_t v)
  {
    check_mem_corruption();
    value = v;
    return *this;
  }

  UserType& operator++()
  {
    check_mem_corruption();
    ++value;
    return *this;
  }

  bool operator==(const UserType& other) const
  {
    check_mem_corruption();
    other.check_mem_corruption();
    return before_padding == other.before_padding && value == other.value &&
           after_padding == other.after_padding;
  }

  void check_mem_corruption() const
  {
    // to test for corruption
    for (auto&& v : before_padding) {
      ASSERT_EQ(v, 0);
    }
    for (auto&& v : after_padding) {
      ASSERT_EQ(v, 0);
    }
  }
};

#define SHARED_PTR_UTIL_FN_BEGIN(...) SCOPED_TRACE(__VA_ARGS__)

template <template <typename> typename SharedPtrImpl, typename T, typename U = T>
inline void test_basic_equal(SharedPtrImpl<T>& ptr, U* bare_ptr, std::size_t N = 0)
{
  SHARED_PTR_UTIL_FN_BEGIN("");
  ASSERT_EQ(ptr.get(), bare_ptr);
  if (bare_ptr) {
    EXPECT_TRUE(ptr);
    ASSERT_TRUE(ptr.get());
    ASSERT_EQ(*ptr, *bare_ptr);
    ASSERT_GT(ptr.use_count(), 0);

    // can't really do this if they are not the same, since otherwise we might slice classes
    if constexpr (std::is_same_v<T, U>) {
      const auto val_before = *bare_ptr;

      ++(*bare_ptr);
      ASSERT_EQ(*ptr, *bare_ptr);
      *ptr = val_before;
      ASSERT_EQ(*ptr, *bare_ptr);

      if constexpr (std::is_array_v<T>) {
        for (std::size_t i = 0; i < N; ++i) {
          ASSERT_EQ(ptr[i], bare_ptr[i]);
          // ensure const overload is also equal
          ASSERT_EQ(const_cast<const SharedPtrImpl<T>&>(ptr)[i], bare_ptr[i]);
        }
      }
    }
  } else {
    ASSERT_FALSE(ptr);
    ASSERT_FALSE(ptr.get());
    ASSERT_EQ(ptr.use_count(), 0);
  }
}

struct Tag {};

template <template <typename> typename SharedPtrImpl, typename T>
inline void test_create_with_copy_n(std::vector<std::reference_wrapper<SharedPtrImpl<T>>> ptrs,
                                    T* bare_ptr,
                                    Tag)
{
  SHARED_PTR_UTIL_FN_BEGIN("");
  ASSERT_TRUE(bare_ptr);
  for (std::size_t i = 0; i < ptrs.size(); ++i) {
    ASSERT_EQ(ptrs[i].get().use_count(), ptrs.size()) << "idx: " << i;
  }
  for (std::size_t i = 0; i < ptrs.size(); ++i) {
    ASSERT_TRUE(ptrs[i].get()) << "idx: " << i;
  }
  for (std::size_t i = 0; i < ptrs.size(); ++i) {
    ASSERT_EQ(ptrs[i].get().get(), bare_ptr) << "idx: " << i;
  }
  for (std::size_t i = 0; i < ptrs.size(); ++i) {
    ASSERT_EQ(*(ptrs[i].get()), *bare_ptr) << "idx: " << i;
  }
}

template <typename T>
inline void test_create_with_copy_n(std::vector<std::reference_wrapper<legate::SharedPtr<T>>> ptrs,
                                    T* bare_ptr)
{
  SHARED_PTR_UTIL_FN_BEGIN("");
  test_create_with_copy_n(std::move(ptrs), bare_ptr, Tag{});
}

template <typename T>
inline void test_create_with_copy_n(
  std::vector<std::reference_wrapper<legate::InternalSharedPtr<T>>> ptrs, T* bare_ptr)
{
  SHARED_PTR_UTIL_FN_BEGIN("");
  test_create_with_copy_n(std::move(ptrs), bare_ptr, Tag{});
}

// clang-tidy emits a new use-after-move diagnostic when using a class after it has been moved
// from. The only two special cases are for std::unique_ptr and std::shared_ptr, which have
// clearly defined moved-from states.
//
// However, clang-tidy does not realize that SharedPtr and InternalSharedPtr have the exact
// same semantics as std::shared_ptr, and hence it will warn if these objects are used after a
// move.
//
// Calling this function on either, after they were moved, will make clang-tidy think the
// object is "re-initialized". See
// https://clang.llvm.org/extra/clang-tidy/checks/bugprone/use-after-move.html#silencing-erroneous-warnings
template <typename T>
void silence_spurious_clang_tidy_use_after_move(T&)
{
}

struct Base {
  std::array<char, 128> padding{};
  std::int32_t value{};

  Base(std::int32_t v) : value{v} {}

  Base()                       = default;
  Base(const Base&)            = default;
  Base& operator=(const Base&) = default;
  virtual ~Base()              = default;

  Base& operator++()
  {
    ++value;
    return *this;
  }

  bool operator==(const Base& other) const
  {
    check_mem_corruption();
    other.check_mem_corruption();
    return padding == other.padding && value == other.value;
  }

  virtual void check_mem_corruption() const
  {
    for (auto&& v : padding) {
      ASSERT_EQ(v, 0);
    }
  }
};

struct BasicDerived : Base {
  std::array<char, 256> more_padding{};

  using Base::Base;

  BasicDerived() = default;

  bool operator==(const BasicDerived& other) const
  {
    return Base::operator==(other) && more_padding == other.more_padding;
  }

  void check_mem_corruption() const override
  {
    // to test for corruption
    Base::check_mem_corruption();
    for (auto&& v : more_padding) {
      ASSERT_EQ(v, 0);
    }
  }
};

struct TogglingDerived final : BasicDerived {
  bool* toggle{};

  TogglingDerived() = delete;
  explicit TogglingDerived(bool* tptr) : toggle{tptr} { check_mem_corruption(); }

  ~TogglingDerived() override { *toggle = true; }

  bool operator==(const TogglingDerived& other) const
  {
    check_mem_corruption();
    return BasicDerived::operator==(other) && *toggle == *other.toggle;
  }

  void check_mem_corruption() const override
  {
    BasicDerived::check_mem_corruption();
    ASSERT_TRUE(toggle);
  }
};

using BasicSharedPtrTypeList = ::testing::Types<std::int8_t, std::int32_t, std::uint64_t, UserType>;

template <typename>
struct BasicSharedPtrUnit : ::testing::Test {};
