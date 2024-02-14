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

#include "core/partitioning/detail/constraint.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace unit {

using Variable        = DefaultFixture;
using Alignment       = DefaultFixture;
using Broadcast       = DefaultFixture;
using ImageConstraint = DefaultFixture;

static const char* library_name = "test_constraints";

enum TaskIDs {
  INIT = 0,
};

// Dummy task to make the runtime think the store is initialized
struct Initializer : public legate::LegateTask<Initializer> {
  static const std::int32_t TASK_ID = INIT;
  static void cpu_variant(legate::TaskContext /*context*/) {}
};

void register_tasks()
{
  static bool prepared = false;
  if (prepared) {
    return;
  }
  prepared     = true;
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->create_library(library_name);
  Initializer::register_variants(context);
}

TEST_F(Variable, BasicMethods)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, INIT);

  // Test basic properties
  auto part     = task.declare_partition();
  auto part_imp = part.impl();
  EXPECT_FALSE(part_imp->closed());
  EXPECT_EQ(part_imp->kind(), legate::detail::Expr::Kind::VARIABLE);
  EXPECT_EQ(part_imp->as_literal(), nullptr);
  EXPECT_EQ(part_imp->as_variable(), part_imp);
  EXPECT_TRUE(part_imp->operation() != nullptr);

  // Test equal
  auto part1(part);
  auto part1_imp = part1.impl();
  EXPECT_EQ(*part_imp, *part1_imp);
  auto part2     = task.declare_partition();
  auto part2_imp = part2.impl();

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  part_imp->find_partition_symbols(symbols);
  part1_imp->find_partition_symbols(symbols);
  part2_imp->find_partition_symbols(symbols);
  EXPECT_EQ(symbols.size(), 3);
  EXPECT_TRUE(std::find(symbols.begin(), symbols.end(), part_imp) != symbols.end());
  EXPECT_TRUE(std::find(symbols.begin(), symbols.end(), part1_imp) != symbols.end());
  EXPECT_TRUE(std::find(symbols.begin(), symbols.end(), part2_imp) != symbols.end());
}

TEST_F(Alignment, BasicMethods)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, INIT);

  auto part1 = task.declare_partition();
  auto part2 = task.declare_partition();

  auto aligment = legate::detail::align(part1.impl(), part2.impl());
  EXPECT_EQ(aligment->kind(), legate::detail::Constraint::Kind::ALIGNMENT);
  EXPECT_EQ(aligment->lhs(), part1.impl());
  EXPECT_EQ(aligment->rhs(), part2.impl());
  EXPECT_EQ(aligment->as_alignment(), aligment.get());
  EXPECT_EQ(aligment->as_broadcast(), nullptr);
  EXPECT_EQ(aligment->as_image_constraint(), nullptr);
  EXPECT_FALSE(aligment->is_trivial());

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  aligment->find_partition_symbols(symbols);
  EXPECT_EQ(symbols.size(), 2);
  EXPECT_TRUE(std::find(symbols.begin(), symbols.end(), part1.impl()) != symbols.end());
  EXPECT_TRUE(std::find(symbols.begin(), symbols.end(), part2.impl()) != symbols.end());
}

TEST_F(Broadcast, BasicMethods)
{
  register_tasks();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(library_name);
  auto task    = runtime->create_task(context, INIT);
  auto part1   = task.declare_partition();

  auto dims      = legate::from_range<std::uint32_t>(3);
  auto broadcast = legate::detail::broadcast(part1.impl(), dims);
  EXPECT_EQ(broadcast->kind(), legate::detail::Constraint::Kind::BROADCAST);
  EXPECT_EQ(broadcast->variable(), part1.impl());
  EXPECT_EQ(broadcast->axes(), dims);
  EXPECT_EQ(broadcast->as_alignment(), nullptr);
  EXPECT_EQ(broadcast->as_broadcast(), broadcast.get());
  EXPECT_EQ(broadcast->as_image_constraint(), nullptr);

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  broadcast->find_partition_symbols(symbols);
  EXPECT_EQ(symbols.size(), 1);
  EXPECT_TRUE(std::find(symbols.begin(), symbols.end(), part1.impl()) != symbols.end());
}

TEST_F(ImageConstraint, BasicMethods)
{
  register_tasks();

  auto runtime    = legate::Runtime::get_runtime();
  auto context    = runtime->find_library(library_name);
  auto task       = runtime->create_task(context, INIT);
  auto part_func  = task.declare_partition();
  auto part_range = task.declare_partition();

  auto image_constraint = legate::detail::image(part_func.impl(), part_range.impl());
  EXPECT_EQ(image_constraint->kind(), legate::detail::Constraint::Kind::IMAGE);
  EXPECT_EQ(image_constraint->var_function(), part_func.impl());
  EXPECT_EQ(image_constraint->var_range(), part_range.impl());
  EXPECT_EQ(image_constraint->as_alignment(), nullptr);
  EXPECT_EQ(image_constraint->as_broadcast(), nullptr);
  EXPECT_EQ(image_constraint->as_image_constraint(), image_constraint.get());

  // Test find_partition_symbols
  std::vector<const legate::detail::Variable*> symbols = {};
  image_constraint->find_partition_symbols(symbols);
  EXPECT_EQ(symbols.size(), 2);
  EXPECT_TRUE(std::find(symbols.begin(), symbols.end(), part_func.impl()) != symbols.end());
  EXPECT_TRUE(std::find(symbols.begin(), symbols.end(), part_range.impl()) != symbols.end());
}
}  // namespace unit
