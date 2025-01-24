/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/partitioning/detail/constraint.h>

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace {

// NOLINTBEGIN(readability-magic-numbers)

// Dummy task to make the runtime think the store is initialized
struct Initializer : public legate::LegateTask<Initializer> {
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_constraints";
  static void registration_callback(legate::Library library)
  {
    Initializer::register_variants(library);
  }
};

class Constraint : public RegisterOnceFixture<Config> {};

TEST_F(Constraint, Variable)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, Initializer::TASK_ID);

  // Test basic properties
  auto part     = task.declare_partition();
  auto part_imp = part.impl();
  ASSERT_FALSE(part_imp->closed());
  ASSERT_TRUE(part_imp->operation() != nullptr);
  ASSERT_FALSE(part.to_string().empty());

  // Test equal
  auto part1(part);
  auto part1_imp = part1.impl();
  ASSERT_EQ(*part_imp, *part1_imp);
  auto part2     = task.declare_partition();
  auto part2_imp = part2.impl();

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  part_imp->find_partition_symbols(symbols);
  part1_imp->find_partition_symbols(symbols);
  part2_imp->find_partition_symbols(symbols);
  ASSERT_EQ(symbols.size(), 3);
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part_imp) != symbols.end());
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part1_imp) != symbols.end());
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part2_imp) != symbols.end());
}

TEST_F(Constraint, Alignment)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, Initializer::TASK_ID);

  auto part1 = task.declare_partition();
  auto part2 = task.declare_partition();

  auto aligment = legate::detail::align(part1.impl(), part2.impl());
  ASSERT_EQ(aligment->kind(), legate::detail::Constraint::Kind::ALIGNMENT);
  ASSERT_EQ(aligment->lhs(), part1.impl());
  ASSERT_EQ(aligment->rhs(), part2.impl());
  ASSERT_EQ(dynamic_cast<const legate::detail::Alignment*>(aligment.get()), aligment.get());
  ASSERT_EQ(dynamic_cast<const legate::detail::Broadcast*>(aligment.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::ImageConstraint*>(aligment.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::ScaleConstraint*>(aligment.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::BloatConstraint*>(aligment.get()), nullptr);
  ASSERT_FALSE(aligment->is_trivial());

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  aligment->find_partition_symbols(symbols);
  ASSERT_EQ(symbols.size(), 2);
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part1.impl()) != symbols.end());
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part2.impl()) != symbols.end());
}

TEST_F(Constraint, Broadcast)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, Initializer::TASK_ID);
  auto part1   = task.declare_partition();

  auto dims      = legate::from_range<std::uint32_t>(3);
  auto broadcast = legate::detail::broadcast(part1.impl(), dims);
  ASSERT_EQ(broadcast->kind(), legate::detail::Constraint::Kind::BROADCAST);
  ASSERT_EQ(broadcast->variable(), part1.impl());
  ASSERT_EQ(broadcast->axes(), dims);
  ASSERT_EQ(dynamic_cast<const legate::detail::Alignment*>(broadcast.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::Broadcast*>(broadcast.get()), broadcast.get());
  ASSERT_EQ(dynamic_cast<const legate::detail::ImageConstraint*>(broadcast.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::ScaleConstraint*>(broadcast.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::BloatConstraint*>(broadcast.get()), nullptr);

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  broadcast->find_partition_symbols(symbols);
  ASSERT_EQ(symbols.size(), 1);
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part1.impl()) != symbols.end());
}

TEST_F(Constraint, ImageConstraint)
{
  auto runtime    = legate::Runtime::get_runtime();
  auto context    = runtime->find_library(Config::LIBRARY_NAME);
  auto task       = runtime->create_task(context, Initializer::TASK_ID);
  auto part_func  = task.declare_partition();
  auto part_range = task.declare_partition();

  auto image_constraint = legate::detail::image(
    part_func.impl(), part_range.impl(), legate::ImageComputationHint::NO_HINT);
  ASSERT_EQ(image_constraint->kind(), legate::detail::Constraint::Kind::IMAGE);
  ASSERT_EQ(image_constraint->var_function(), part_func.impl());
  ASSERT_EQ(image_constraint->var_range(), part_range.impl());
  ASSERT_EQ(dynamic_cast<const legate::detail::Alignment*>(image_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::Broadcast*>(image_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::ImageConstraint*>(image_constraint.get()),
            image_constraint.get());
  ASSERT_EQ(dynamic_cast<const legate::detail::ScaleConstraint*>(image_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::BloatConstraint*>(image_constraint.get()), nullptr);

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  image_constraint->find_partition_symbols(symbols);
  ASSERT_EQ(symbols.size(), 2);
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part_func.impl()) != symbols.end());
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part_range.impl()) != symbols.end());
}

TEST_F(Constraint, ScaleConstraint)
{
  auto runtime      = legate::Runtime::get_runtime();
  auto context      = runtime->find_library(Config::LIBRARY_NAME);
  auto task         = runtime->create_task(context, Initializer::TASK_ID);
  auto smaller      = runtime->create_store({3}, legate::int64());
  auto bigger       = runtime->create_store({5}, legate::int64());
  auto part_smaller = task.add_output(smaller);
  auto part_bigger  = task.add_output(bigger);

  auto scale_constraint = legate::detail::scale({1}, part_smaller.impl(), part_bigger.impl());
  ASSERT_EQ(scale_constraint->kind(), legate::detail::Constraint::Kind::SCALE);
  ASSERT_EQ(scale_constraint->var_smaller(), part_smaller.impl());
  ASSERT_EQ(scale_constraint->var_bigger(), part_bigger.impl());
  ASSERT_EQ(dynamic_cast<const legate::detail::Alignment*>(scale_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::Broadcast*>(scale_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::ImageConstraint*>(scale_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::ScaleConstraint*>(scale_constraint.get()),
            scale_constraint.get());
  ASSERT_EQ(dynamic_cast<const legate::detail::BloatConstraint*>(scale_constraint.get()), nullptr);

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  scale_constraint->find_partition_symbols(symbols);
  ASSERT_EQ(symbols.size(), 2);
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part_smaller.impl()) != symbols.end());
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part_bigger.impl()) != symbols.end());
}

TEST_F(Constraint, BloatConstraint)
{
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, Initializer::TASK_ID);
  auto source  = runtime->create_store({5}, legate::int64());
  auto bloated = runtime->create_store({5}, legate::int64());
  runtime->issue_fill(source, legate::Scalar(std::int64_t{0}));
  runtime->issue_fill(bloated, legate::Scalar(std::int64_t{0}));
  auto part_source  = task.add_input(source);
  auto part_bloated = task.add_input(bloated);

  auto bloat_constraint = legate::detail::bloat(part_source.impl(), part_bloated.impl(), {1}, {3});
  ASSERT_EQ(bloat_constraint->kind(), legate::detail::Constraint::Kind::BLOAT);
  ASSERT_EQ(bloat_constraint->var_source(), part_source.impl());
  ASSERT_EQ(bloat_constraint->var_bloat(), part_bloated.impl());
  ASSERT_EQ(dynamic_cast<const legate::detail::Alignment*>(bloat_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::Broadcast*>(bloat_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::ImageConstraint*>(bloat_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::ScaleConstraint*>(bloat_constraint.get()), nullptr);
  ASSERT_EQ(dynamic_cast<const legate::detail::BloatConstraint*>(bloat_constraint.get()),
            bloat_constraint.get());

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  bloat_constraint->find_partition_symbols(symbols);
  ASSERT_EQ(symbols.size(), 2);
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part_source.impl()) != symbols.end());
  ASSERT_TRUE(std::find(symbols.begin(), symbols.end(), part_bloated.impl()) != symbols.end());
}

// NOLINTEND(readability-magic-numbers)
}  // namespace
