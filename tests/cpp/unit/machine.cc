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

#include <legate/mapping/detail/machine.h>

#include <legate.h>

#include <legate/mapping/machine.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/deserializer.h>

#include <gtest/gtest.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <utilities/utilities.h>
#include <valarray>

namespace unit {

using MachineTest        = DefaultFixture;
using NodeRangeTest      = DefaultFixture;
using ProcessorRangeTest = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr legate::mapping::ProcessorRange CPU_RANGE{1, 3, 4};
constexpr legate::mapping::ProcessorRange OMP_RANGE{0, 3, 2};
constexpr legate::mapping::ProcessorRange GPU_RANGE{3, 6, 3};

[[nodiscard]] bool check_task_target_vec(std::vector<legate::mapping::TaskTarget> input,
                                         std::vector<legate::mapping::TaskTarget> expect)
{
  std::sort(input.begin(), input.end());
  std::sort(expect.begin(), expect.end());

  return input == expect;
}

}  // namespace

TEST_F(NodeRangeTest, ComparisonOperators)
{
  constexpr legate::mapping::NodeRange range1{1, 3};
  constexpr legate::mapping::NodeRange range2{2, 3};
  constexpr legate::mapping::NodeRange range3{1, 4};

  // Test NodeRange operators
  static_assert(range1 < range2);
  static_assert(range1 < range3);
  static_assert(range1 != range2);
  static_assert(range1 != range3);
  static_assert(!(range1 == range2));
  static_assert(!(range1 == range3));
}

TEST_F(ProcessorRangeTest, Create)
{
  constexpr legate::mapping::ProcessorRange range{1, 3, 1};

  static_assert(!range.empty());
  static_assert(range.per_node_count == 1);
  static_assert(range.low == 1);
  static_assert(range.high == 3);
  static_assert(range.count() == 2);
  static_assert(range.get_node_range() == legate::mapping::NodeRange{1, 3});
}

TEST_F(ProcessorRangeTest, CreateDefault)
{
  constexpr legate::mapping::ProcessorRange range;

  static_assert(range.empty());
  static_assert(range.per_node_count == 1);
  static_assert(range.low == 0);
  static_assert(range.high == 0);
  static_assert(range.count() == 0);
}

TEST_F(ProcessorRangeTest, CreateEmpty)
{
  constexpr auto check_empty = [](const legate::mapping::ProcessorRange& range) {
    ASSERT_TRUE(range.empty());
    ASSERT_EQ(range.per_node_count, 1);
    ASSERT_EQ(range.low, 0);
    ASSERT_EQ(range.high, 0);
    ASSERT_EQ(range.count(), 0);
    ASSERT_THROW(static_cast<void>(range.get_node_range()), std::invalid_argument);
  };

  constexpr legate::mapping::ProcessorRange range1{1, 0, 1};
  check_empty(range1);

  constexpr legate::mapping::ProcessorRange range2{3, 3, 0};
  check_empty(range2);
}

TEST_F(ProcessorRangeTest, ComparisonOperator)
{
  constexpr legate::mapping::ProcessorRange range1{2, 6, 2};
  constexpr legate::mapping::ProcessorRange range2{2, 6, 2};
  static_assert(range1 == range2);

  constexpr legate::mapping::ProcessorRange range3{1, 6, 2};
  static_assert(range1 != range3);
  static_assert(range3 < range1);
  static_assert(!(range1 < range3));

  constexpr legate::mapping::ProcessorRange range4{2, 5, 2};
  static_assert(range4 < range1);
  static_assert(!(range1 < range4));

  constexpr legate::mapping::ProcessorRange range5{2, 6, 1};
  static_assert(range5 < range1);
  static_assert(!(range1 < range5));
}

TEST_F(ProcessorRangeTest, IntersectionOperator)
{
  // Generate nonempty range
  constexpr legate::mapping::ProcessorRange range1{0, 3, 1};
  constexpr legate::mapping::ProcessorRange range2{2, 4, 1};
  constexpr auto result1 = range1 & range2;
  static_assert(result1 == legate::mapping::ProcessorRange{2, 3, 1});

  // Generate empty range
  constexpr legate::mapping::ProcessorRange range3{0, 2, 1};
  constexpr legate::mapping::ProcessorRange range4{3, 5, 1};
  constexpr auto result2 = range3 & range4;
  static_assert(result2 == legate::mapping::ProcessorRange{0, 0, 1});
  static_assert(result2.count() == 0);

  // Throw exception
  constexpr legate::mapping::ProcessorRange range5{1, 3, 1};
  constexpr legate::mapping::ProcessorRange range6{2, 4, 2};
  ASSERT_THROW(static_cast<void>(range5 & range6), std::invalid_argument);
}

TEST_F(ProcessorRangeTest, NodeRange)
{
  constexpr legate::mapping::ProcessorRange range{0, 7, 2};
  static_assert(range.get_node_range() == legate::mapping::NodeRange{0, 4});
}

TEST_F(ProcessorRangeTest, Slice)
{
  // Slice empty range with empty range
  constexpr legate::mapping::ProcessorRange range1{3, 1, 1};
  static_assert(range1.slice(0, 0).count() == 0);
  static_assert(range1.slice(4, 6).count() == 0);

  // Slice nonempty range with empty range
  constexpr legate::mapping::ProcessorRange range2{1, 3, 1};
  static_assert(range2.slice(0, 0).count() == 0);
  static_assert(range2.slice(4, 6).count() == 0);
  static_assert(range2.slice(1, 0).count() == 0);

  // Slice nonempty range with nonempty range
  constexpr legate::mapping::ProcessorRange range3{1, 3, 1};
  static_assert(range3.slice(1, 3).count() == 1);
  static_assert(range3.slice(0, 2).count() == 2);
}

TEST_F(ProcessorRangeTest, ToString)
{
  constexpr legate::mapping::ProcessorRange range{1, 3, 1};
  constexpr std::string_view range_str = "Proc([1,3], 1 per node)";

  std::stringstream ss;
  ss << range;
  ASSERT_EQ(ss.str(), range_str);
  ASSERT_EQ(range.to_string(), range_str);
}

TEST_F(MachineTest, EmptyMachine)
{
  const std::map<legate::mapping::TaskTarget, legate::mapping::ProcessorRange> processor_ranges = {
    {legate::mapping::TaskTarget::CPU, {0, 0, 1}}};
  const legate::mapping::Machine machine{processor_ranges};

  ASSERT_EQ(machine.preferred_target(), legate::mapping::TaskTarget::CPU);
  ASSERT_EQ(machine.count(), 0);
  ASSERT_EQ(machine.count(legate::mapping::TaskTarget::GPU), 0);
  ASSERT_EQ(machine.processor_range(), (legate::mapping::ProcessorRange{0, 0, 1}));
  ASSERT_EQ(machine.processor_range(legate::mapping::TaskTarget::GPU),
            (legate::mapping::ProcessorRange{0, 0, 1}));
  ASSERT_EQ(machine.slice(0, 1),
            (legate::mapping::Machine{
              {{legate::mapping::TaskTarget::CPU, legate::mapping::ProcessorRange{}}}}));
  ASSERT_TRUE(machine.empty());
  ASSERT_EQ(machine.valid_targets().size(), 0);
  ASSERT_EQ(machine.only(legate::mapping::TaskTarget::CPU),
            (legate::mapping::Machine{
              {{legate::mapping::TaskTarget::CPU, legate::mapping::ProcessorRange{}}}}));
  ASSERT_EQ(machine.impl()->processor_ranges(), processor_ranges);
}

TEST_F(MachineTest, EqualOperator)
{
  const legate::mapping::Machine machine1{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::OMP, OMP_RANGE}}};
  const legate::mapping::Machine machine2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::OMP, OMP_RANGE}}};
  ASSERT_EQ(machine1, machine2);

  const legate::mapping::Machine machine3{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  ASSERT_NE(machine3, machine1);
}

TEST_F(MachineTest, PreferedTarget)
{
  const legate::mapping::Machine machine1{{{legate::mapping::TaskTarget::CPU, CPU_RANGE}}};
  ASSERT_EQ(machine1.preferred_target(), legate::mapping::TaskTarget::CPU);

  const legate::mapping::Machine machine2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::OMP, OMP_RANGE}}};
  ASSERT_EQ(machine2.preferred_target(), legate::mapping::TaskTarget::OMP);

  const legate::mapping::Machine machine3{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                           {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                           {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  ASSERT_EQ(machine3.preferred_target(), legate::mapping::TaskTarget::GPU);
}

TEST_F(MachineTest, ProcessorRange)
{
  const legate::mapping::Machine machine1{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                           {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                           {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  ASSERT_EQ(machine1.processor_range(), GPU_RANGE);
  ASSERT_EQ(machine1.processor_range(legate::mapping::TaskTarget::CPU), CPU_RANGE);
  ASSERT_EQ(machine1.processor_range(legate::mapping::TaskTarget::OMP), OMP_RANGE);
  ASSERT_EQ(machine1.processor_range(legate::mapping::TaskTarget::GPU), GPU_RANGE);

  const legate::mapping::Machine machine2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::OMP, OMP_RANGE}}};

  ASSERT_EQ(machine2.processor_range(), OMP_RANGE);
  ASSERT_EQ(machine2.processor_range(legate::mapping::TaskTarget::CPU), CPU_RANGE);
  ASSERT_EQ(machine2.processor_range(legate::mapping::TaskTarget::GPU),
            legate::mapping::ProcessorRange{});
}

TEST_F(MachineTest, ValidTargets)
{
  const legate::mapping::Machine machine1{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                           {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                           {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const auto& valid_targets1 = machine1.valid_targets();
  const auto targets1 = std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU,
                                                                 legate::mapping::TaskTarget::OMP,
                                                                 legate::mapping::TaskTarget::GPU};
  ASSERT_TRUE(check_task_target_vec(valid_targets1, targets1));

  const legate::mapping::Machine machine2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::OMP, OMP_RANGE}}};
  const auto& valid_targets2 = machine2.valid_targets();
  const auto targets2 = std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU,
                                                                 legate::mapping::TaskTarget::OMP};
  ASSERT_TRUE(check_task_target_vec(valid_targets2, targets2));
}

TEST_F(MachineTest, ValidTargetsExcept)
{
  const legate::mapping::Machine machine{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                          {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                          {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  std::set<legate::mapping::TaskTarget> exclude_targets;
  const auto valid_targets1 = machine.valid_targets_except(exclude_targets);
  const auto targets1 = std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU,
                                                                 legate::mapping::TaskTarget::OMP,
                                                                 legate::mapping::TaskTarget::GPU};
  ASSERT_TRUE(check_task_target_vec(valid_targets1, targets1));

  exclude_targets.insert(legate::mapping::TaskTarget::CPU);
  const auto valid_targets2 = machine.valid_targets_except(exclude_targets);
  const auto targets2 = std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::OMP,
                                                                 legate::mapping::TaskTarget::GPU};
  ASSERT_TRUE(check_task_target_vec(valid_targets2, targets2));

  exclude_targets.insert(legate::mapping::TaskTarget::OMP);
  const auto valid_targets3 = machine.valid_targets_except(exclude_targets);
  const auto targets3 = std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::GPU};
  ASSERT_TRUE(check_task_target_vec(valid_targets3, targets3));

  exclude_targets.insert(legate::mapping::TaskTarget::GPU);
  const auto valid_targets4 = machine.valid_targets_except(exclude_targets);
  ASSERT_EQ(valid_targets4.size(), 0);
}

TEST_F(MachineTest, Count)
{
  const legate::mapping::Machine machine{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                          {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                          {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  ASSERT_EQ(machine.count(), 3);
  ASSERT_EQ(machine.count(legate::mapping::TaskTarget::CPU), 2);
  ASSERT_EQ(machine.count(legate::mapping::TaskTarget::OMP), 3);
}

TEST_F(MachineTest, Only)
{
  const legate::mapping::Machine machine{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const auto machine1        = machine.only(legate::mapping::TaskTarget::CPU);
  const auto& valid_targets1 = machine1.valid_targets();
  ASSERT_EQ(valid_targets1,
            std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU});
  ASSERT_EQ(machine1.count(), 2);
  ASSERT_EQ(machine1.preferred_target(), legate::mapping::TaskTarget::CPU);
  ASSERT_EQ(machine1.processor_range().per_node_count, 4);

  const legate::mapping::Machine machine2{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                           {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                           {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const auto machine3 =
    machine2.only({legate::mapping::TaskTarget::GPU, legate::mapping::TaskTarget::CPU});
  ASSERT_EQ(machine3, machine);
}

TEST_F(MachineTest, OnlyIf)
{
  const legate::mapping::Machine machine{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                          {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                          {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  auto machine1 = machine.impl()->only_if(
    [](legate::mapping::TaskTarget t) { return t == legate::mapping::TaskTarget::CPU; });
  ASSERT_EQ(machine1.valid_targets(),
            std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU});
  ASSERT_EQ(machine1.preferred_target(), legate::mapping::TaskTarget::CPU);
}

TEST_F(MachineTest, Slice)
{
  const legate::mapping::Machine machine1{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const legate::mapping::Machine expected{
    {{legate::mapping::TaskTarget::GPU, GPU_RANGE.slice(0, 1)}}};

  ASSERT_EQ(machine1.slice(0, 1), expected);

  const auto new_machine1 = machine1.slice(0, 2, legate::mapping::TaskTarget::GPU);

  ASSERT_EQ(new_machine1.preferred_target(), legate::mapping::TaskTarget::GPU);
  ASSERT_EQ(new_machine1.processor_range().count(), 2);

  const legate::mapping::Machine machine2{{{legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const auto new_machine2 = machine2.slice(0, 2);

  ASSERT_EQ(new_machine2.preferred_target(), legate::mapping::TaskTarget::GPU);
  ASSERT_EQ(new_machine2.processor_range().count(), 2);

  const legate::mapping::Machine machine3{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                           {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                           {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const legate::mapping::Machine expected1{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE},
     {legate::mapping::TaskTarget::OMP, OMP_RANGE},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE.slice(0, 1)}}};

  ASSERT_EQ(machine3.slice(0, 1, true), expected1);

  const legate::mapping::Machine expected2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE.slice(1, 2)},
     {legate::mapping::TaskTarget::OMP, OMP_RANGE},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  ASSERT_EQ(machine3.slice(1, 2, legate::mapping::TaskTarget::CPU, true), expected2);
}

TEST_F(MachineTest, IndexOperator)
{
  const legate::mapping::Machine machine{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                          {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                          {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const auto machine1        = machine[legate::mapping::TaskTarget::GPU];
  const auto& valid_targets1 = machine1.valid_targets();

  ASSERT_EQ(valid_targets1,
            std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::GPU});
  ASSERT_EQ(machine1.count(), 3);
  ASSERT_EQ(machine1.preferred_target(), legate::mapping::TaskTarget::GPU);
  ASSERT_EQ(machine1.processor_range().per_node_count, 3);

  const auto targets  = std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU,
                                                                 legate::mapping::TaskTarget::OMP};
  const auto machine2 = machine[targets];
  const auto& valid_targets2 = machine2.valid_targets();

  ASSERT_TRUE(check_task_target_vec(valid_targets2, targets));
  ASSERT_EQ(machine2.preferred_target(), legate::mapping::TaskTarget::OMP);
  ASSERT_EQ(machine2.processor_range().per_node_count, 2);
}

TEST_F(MachineTest, IntersectionOperator)
{
  const legate::mapping::Machine machine1{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE.slice(1, 2)},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  const legate::mapping::Machine machine2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE},
     {legate::mapping::TaskTarget::OMP, OMP_RANGE},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE.slice(0, 1)}}};

  const legate::mapping::Machine machine3{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE.slice(1, 2)},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE.slice(0, 1)}}};
  ASSERT_EQ(machine3, (machine1 & machine2));

  const legate::mapping::Machine machine4{{{legate::mapping::TaskTarget::CPU, {0, 0, 1}}}};
  ASSERT_EQ(machine2.only(legate::mapping::TaskTarget::OMP) & machine1, machine4);
}

TEST_F(MachineTest, ToString)
{
  const legate::mapping::Machine machine{{{legate::mapping::TaskTarget::CPU, CPU_RANGE}}};
  constexpr std::string_view machine_str =
    "Machine(preferred_target: CPU, CPU: Proc([1,3], 4 per node))";

  std::stringstream ss;
  ss << machine;
  ASSERT_EQ(ss.str(), machine_str);
  ASSERT_EQ(machine.to_string(), machine_str);
}

class MachineUnitTestDeserializer
  : public legate::detail::BaseDeserializer<MachineUnitTestDeserializer> {
 public:
  MachineUnitTestDeserializer(const void* args, std::size_t arglen) : BaseDeserializer{args, arglen}
  {
  }

  using BaseDeserializer::unpack_impl;
};

TEST_F(MachineTest, Pack)
{
  legate::detail::BufferBuilder buf;
  const legate::mapping::detail::Machine machine{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  // Copy is intentional because the const object may also be changed in mutable functions
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const auto orig_machine_copy = machine;
  machine.pack(buf);
  auto legion_buffer = buf.to_legion_buffer();
  MachineUnitTestDeserializer dez{legion_buffer.get_ptr(), legion_buffer.get_size()};

  auto machine_unpack = dez.unpack<legate::mapping::detail::Machine>();

  ASSERT_EQ(machine_unpack, orig_machine_copy);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace unit
