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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace scope_test {

using ScopeTest = DefaultFixture;

namespace {

constexpr std::int32_t MAGIC_PRIORITY1 = 42;
constexpr std::int32_t MAGIC_PRIORITY2 = 43;

const std::string MAGIC_PROVENANCE1 = "42";
const std::string MAGIC_PROVENANCE2 = "43";

}  // namespace

legate::mapping::Machine remove_last_proc(legate::mapping::Machine machine)
{
  return machine.slice(0, std::max(std::uint32_t{1}, machine.count() - 1));
}

TEST_F(ScopeTest, BasicPriority)
{
  const auto old_priority = legate::Scope::priority();
  {
    legate::Scope test_priority{MAGIC_PRIORITY1};
    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
}

TEST_F(ScopeTest, BasicProvenance)
{
  const auto old_provenance = legate::Scope::provenance();
  {
    legate::Scope test_provenance{MAGIC_PROVENANCE1};
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
  }
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
}

TEST_F(ScopeTest, BasicMachine)
{
  const auto old_machine = legate::Scope::machine();
  {
    auto sliced = remove_last_proc(legate::Scope::machine());
    legate::Scope test_machine{sliced};
    EXPECT_EQ(legate::Scope::machine(), sliced);
  }
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, NestedPriority)
{
  const auto old_priority = legate::Scope::priority();
  {
    legate::Scope test_priority1{MAGIC_PRIORITY1};
    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);

    {
      legate::Scope test_priority2{MAGIC_PRIORITY2};
      EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY2);
    }

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
}

TEST_F(ScopeTest, NestedProvenance)
{
  const auto old_provenance = legate::Scope::provenance();
  {
    legate::Scope test_provenance1{MAGIC_PROVENANCE1};
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);

    {
      legate::Scope test_provenance2{MAGIC_PROVENANCE2};
      EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE2);
    }

    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
  }
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
}

TEST_F(ScopeTest, NestedMachine)
{
  const auto old_machine = legate::Scope::machine();
  {
    auto sliced1 = remove_last_proc(legate::Scope::machine());
    legate::Scope test_machine1{sliced1};
    EXPECT_EQ(legate::Scope::machine(), sliced1);

    {
      auto sliced2 = remove_last_proc(legate::Scope::machine());
      legate::Scope test_machine2{sliced2};
      EXPECT_EQ(legate::Scope::machine(), sliced2);
    }

    EXPECT_EQ(legate::Scope::machine(), sliced1);
  }
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, BasicChain)
{
  const auto old_priority   = legate::Scope::priority();
  const auto old_provenance = legate::Scope::provenance();
  const auto old_machine    = legate::Scope::machine();
  {
    auto sliced   = remove_last_proc(legate::Scope::machine());
    auto test_all = legate::Scope{}
                      .with_priority(MAGIC_PRIORITY1)
                      .with_provenance(MAGIC_PROVENANCE1)
                      .with_machine(sliced);
    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, NestedChain)
{
  const auto old_priority   = legate::Scope::priority();
  const auto old_provenance = legate::Scope::provenance();
  const auto old_machine    = legate::Scope::machine();
  {
    auto sliced1   = remove_last_proc(legate::Scope::machine());
    auto test_all1 = legate::Scope{}
                       .with_priority(MAGIC_PRIORITY1)
                       .with_provenance(MAGIC_PROVENANCE1)
                       .with_machine(sliced1);
    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced1);

    {
      auto sliced2   = remove_last_proc(legate::Scope::machine());
      auto test_all2 = legate::Scope{}
                         .with_priority(MAGIC_PRIORITY2)
                         .with_provenance(MAGIC_PROVENANCE2)
                         .with_machine(sliced2);

      EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY2);
      EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE2);
      EXPECT_EQ(legate::Scope::machine(), sliced2);
    }

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced1);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, BasicSet)
{
  const auto old_priority   = legate::Scope::priority();
  const auto old_provenance = legate::Scope::provenance();
  const auto old_machine    = legate::Scope::machine();
  {
    auto sliced   = remove_last_proc(legate::Scope::machine());
    auto test_all = legate::Scope{};
    test_all.set_priority(MAGIC_PRIORITY1);
    test_all.set_provenance(MAGIC_PROVENANCE1);
    test_all.set_machine(sliced);

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, NestedSet)
{
  const auto old_priority   = legate::Scope::priority();
  const auto old_provenance = legate::Scope::provenance();
  const auto old_machine    = legate::Scope::machine();
  {
    auto sliced1   = remove_last_proc(legate::Scope::machine());
    auto test_all1 = legate::Scope{};
    test_all1.set_priority(MAGIC_PRIORITY1);
    test_all1.set_provenance(MAGIC_PROVENANCE1);
    test_all1.set_machine(sliced1);

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced1);

    {
      auto sliced2   = remove_last_proc(legate::Scope::machine());
      auto test_all2 = legate::Scope{};
      test_all2.set_priority(MAGIC_PRIORITY2);
      test_all2.set_provenance(MAGIC_PROVENANCE2);
      test_all2.set_machine(sliced2);

      EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY2);
      EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE2);
      EXPECT_EQ(legate::Scope::machine(), sliced2);
    }

    EXPECT_EQ(legate::Scope::priority(), MAGIC_PRIORITY1);
    EXPECT_EQ(legate::Scope::provenance(), MAGIC_PROVENANCE1);
    EXPECT_EQ(legate::Scope::machine(), sliced1);
  }
  EXPECT_EQ(legate::Scope::priority(), old_priority);
  EXPECT_EQ(legate::Scope::provenance(), old_provenance);
  EXPECT_EQ(legate::Scope::machine(), old_machine);
}

TEST_F(ScopeTest, DuplicatePriority1)
{
  legate::Scope test_priority{};
  test_priority.set_priority(MAGIC_PRIORITY1);
  EXPECT_THROW(test_priority.set_priority(MAGIC_PRIORITY2), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicatePriority2)
{
  legate::Scope test_priority{MAGIC_PRIORITY1};
  EXPECT_THROW(test_priority.set_priority(MAGIC_PRIORITY2), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicatePriority3)
{
  EXPECT_THROW(static_cast<void>(legate::Scope{MAGIC_PRIORITY1}.with_priority(MAGIC_PRIORITY2)),
               std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateProvenance1)
{
  legate::Scope test_provenance{};
  test_provenance.set_provenance(MAGIC_PROVENANCE1);
  EXPECT_THROW(test_provenance.set_provenance(MAGIC_PROVENANCE2), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateProvenance2)
{
  legate::Scope test_provenance{MAGIC_PROVENANCE1};
  EXPECT_THROW(test_provenance.set_provenance(MAGIC_PROVENANCE2), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateProvenance3)
{
  EXPECT_THROW(
    static_cast<void>(legate::Scope{MAGIC_PROVENANCE1}.with_provenance(MAGIC_PROVENANCE2)),
    std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateMachine1)
{
  legate::Scope test_machine{};
  test_machine.set_machine(legate::Scope::machine());
  EXPECT_THROW(test_machine.set_machine(legate::Scope::machine()), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateMachine2)
{
  legate::Scope test_machine{legate::Scope::machine()};
  EXPECT_THROW(test_machine.set_machine(legate::Scope::machine()), std::invalid_argument);
}

TEST_F(ScopeTest, DuplicateMachine3)
{
  EXPECT_THROW(static_cast<void>(
                 legate::Scope{legate::Scope::machine()}.with_machine(legate::Scope::machine())),
               std::invalid_argument);
}

}  // namespace scope_test
