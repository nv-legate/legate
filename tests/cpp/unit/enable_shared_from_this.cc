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
#include <numeric>
#include <unordered_set>
#include <vector>

namespace enable_shared_from_this_tests {

class EnsureDeleted : public legate::EnableSharedFromThis<EnsureDeleted> {
 public:
  static constexpr auto SIZE = 100;

  explicit EnsureDeleted(bool* deleted) : values_(SIZE, 0), deleted_{deleted}
  {
    SCOPED_TRACE("EnsureDeleted::EnsureDeleted()");
    *deleted_ = false;
    std::iota(values_.begin(), values_.end(), 0);
    // Yes, clang-tidy, I know that this won't do an actual virtual call, but we can at least
    // check this class
    validate();  // NOLINT(clang-analyzer-optin.cplusplus.VirtualCall)
  }

  virtual ~EnsureDeleted()
  {
    SCOPED_TRACE("EnsureDeleted::~EnsureDeleted()");
    validate();  // NOLINT(clang-analyzer-optin.cplusplus.VirtualCall)
    *deleted_ = true;
  }

  EnsureDeleted(const EnsureDeleted&)                = default;
  EnsureDeleted(EnsureDeleted&&) noexcept            = default;
  EnsureDeleted& operator=(const EnsureDeleted&)     = default;
  EnsureDeleted& operator=(EnsureDeleted&&) noexcept = default;

  bool operator==(const EnsureDeleted& other) const
  {
    SCOPED_TRACE("EnsureDeleted::operator==()");
    validate();
    other.validate();
    return values_ == other.values_ && *deleted_ == *other.deleted_;
  }

  virtual void validate() const
  {
    SCOPED_TRACE("EnsureDeleted::validate()");
    int i = 0;

    EXPECT_EQ(values_.size(), SIZE);
    for (const auto& v : values_) {
      EXPECT_EQ(v, i);
      ++i;
    }
    EXPECT_NE(deleted_, nullptr);
    EXPECT_FALSE(*deleted_);
  }

 protected:
  std::vector<int> values_{};
  bool* deleted_{};
};

class EnsureDeletedVirtual : public EnsureDeleted {
 public:
  explicit EnsureDeletedVirtual(bool* deleted)
    : EnsureDeleted{deleted}, set_{this->values_.begin(), this->values_.end()}
  {
    SCOPED_TRACE("EnsureDeletedVirtual::EnsureDeletedVirtual()");
    validate();  // NOLINT(clang-analyzer-optin.cplusplus.VirtualCall)
  }

  bool operator==(const EnsureDeletedVirtual& other) const
  {
    SCOPED_TRACE("EnsureDeletedVirtual::operator==()");
    return EnsureDeleted::operator==(other) && set_ == other.set_;
  }

  void validate() const override
  {
    SCOPED_TRACE("EnsureDeletedVirtual::validate()");
    EnsureDeleted::validate();

    std::vector<int> seen(SIZE, 0);

    EXPECT_EQ(set_.size(), SIZE);
    for (auto&& v : set_) {
      EXPECT_GE(v, 0);
      ++seen.at(static_cast<std::size_t>(v));
    }
    for (auto&& s : seen) {
      EXPECT_EQ(s, 1);
    }
  }

 private:
  std::unordered_set<int> set_{};
};

using TypeList = ::testing::
  Types<EnsureDeleted, const EnsureDeleted, EnsureDeletedVirtual, const EnsureDeletedVirtual>;

template <typename>
struct EnableSharedFromThisUnit : ::testing::Test {};

TYPED_TEST_SUITE(EnableSharedFromThisUnit, TypeList, );

TYPED_TEST(EnableSharedFromThisUnit, Create)
{
  bool del = false;
  {
    auto foo_shared = legate::make_internal_shared<TypeParam>(&del);

    foo_shared->validate();
    EXPECT_FALSE(del);
  }
  EXPECT_TRUE(del);
}

TYPED_TEST(EnableSharedFromThisUnit, SharedFromThisFromMakeShared)
{
  bool del = false;
  {
    auto foo_shared = legate::make_internal_shared<TypeParam>(&del);

    foo_shared->validate();
    EXPECT_FALSE(del);

    auto foo2 = foo_shared->shared_from_this();

    foo_shared->validate();
    foo2->validate();
    EXPECT_FALSE(del);
    EXPECT_EQ(foo2, foo_shared);
    EXPECT_EQ(foo2.use_count(), 2);
    EXPECT_EQ(foo_shared.use_count(), 2);
    EXPECT_TRUE(foo2.get());
    EXPECT_TRUE(foo_shared.get());
    EXPECT_EQ(*foo2, *foo_shared);
  }
  EXPECT_TRUE(del);
}

TYPED_TEST(EnableSharedFromThisUnit, SharedFromThisFromNew)
{
  bool del = false;
  {
    auto foo_shared = legate::InternalSharedPtr<TypeParam>{new std::remove_cv_t<TypeParam>{&del}};

    foo_shared->validate();
    EXPECT_FALSE(del);

    auto foo2 = foo_shared->shared_from_this();

    foo_shared->validate();
    foo2->validate();
    EXPECT_FALSE(del);
    EXPECT_EQ(foo2, foo_shared);
    EXPECT_EQ(foo2.use_count(), 2);
    EXPECT_EQ(foo_shared.use_count(), 2);
    EXPECT_TRUE(foo2.get());
    EXPECT_TRUE(foo_shared.get());
    EXPECT_EQ(*foo2, *foo_shared);
  }
  EXPECT_TRUE(del);
}

class EnsureDeletedRecursive : public EnsureDeletedVirtual {
 public:
  explicit EnsureDeletedRecursive(bool* deleted,
                                  legate::InternalSharedPtr<EnsureDeleted> parent = {})
    : EnsureDeletedVirtual{deleted}, parent_{std::move(parent)}
  {
  }

  bool operator==(const EnsureDeletedRecursive& other) const
  {
    SCOPED_TRACE("EnsureDeletedRecursive::operator==()");
    return EnsureDeletedVirtual::operator==(other) && parent_ == other.parent_;
  }

  void validate() const override
  {
    SCOPED_TRACE("EnsureDeletedRecursive::validate()");
    EnsureDeletedVirtual::validate();

    if (parent_) {
      EXPECT_NE(parent_.get(), nullptr);
      EXPECT_GE(parent_.use_count(), 1);
      parent_->validate();
    }
  }

  void validate(bool should_have_parent) const
  {
    SCOPED_TRACE("EnsureDeletedRecursive::validate(should_have_parent)");
    if (should_have_parent) {
      EXPECT_NE(parent_.get(), nullptr);
    } else {
      EXPECT_EQ(parent_.get(), nullptr);
    }
    this->validate();
  }

 private:
  legate::InternalSharedPtr<EnsureDeleted> parent_{};
};

TEST(EnableSharedFromThisUnit, SharedFromThisRecursive)
{
  bool del_outer = false;
  {
    auto parent = legate::make_internal_shared<EnsureDeletedRecursive>(&del_outer);

    parent->validate(false);  // should not have a parent

    bool del_inner = false;
    {
      auto child = legate::make_internal_shared<EnsureDeletedRecursive>(&del_inner,
                                                                        parent->shared_from_this());

      EXPECT_FALSE(del_inner);
      EXPECT_FALSE(del_outer);
      child->validate(true);    // has a parent
      parent->validate(false);  // still doesn't have a parent:(
    }
    EXPECT_TRUE(del_inner);
    EXPECT_FALSE(del_outer);
  }
  EXPECT_TRUE(del_outer);
}

}  // namespace enable_shared_from_this_tests
