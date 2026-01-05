/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/machine.h>

#include <legate.h>

#include <legate/mapping/machine.h>
#include <legate/utilities/detail/buffer_builder.h>
#include <legate/utilities/detail/deserializer.h>

#include <gtest/gtest.h>

#include <sstream>
#include <utilities/utilities.h>

namespace machine_test {

using MachineTest = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr legate::mapping::ProcessorRange CPU_RANGE{
  /*low_id=*/1, /*high_id=*/3, /*per_node_proc_count=*/4};
constexpr legate::mapping::ProcessorRange OMP_RANGE{
  /*low_id=*/0, /*high_id=*/3, /*per_node_proc_count=*/2};
constexpr legate::mapping::ProcessorRange GPU_RANGE{
  /*low_id=*/3, /*high_id=*/6, /*per_node_proc_count=*/3};

[[nodiscard]] bool check_task_target_vec(std::vector<legate::mapping::TaskTarget> input,
                                         std::vector<legate::mapping::TaskTarget> expect)
{
  std::sort(input.begin(), input.end());
  std::sort(expect.begin(), expect.end());

  return input == expect;
}

template <typename T>
[[nodiscard]] std::vector<T> to_vector(legate::Span<const T> span)
{
  return {span.begin(), span.end()};
}

}  // namespace

TEST_F(MachineTest, Create)
{
  const std::map<legate::mapping::TaskTarget, legate::mapping::ProcessorRange> processor_ranges = {
    {legate::mapping::TaskTarget::CPU, CPU_RANGE}};
  const legate::mapping::Machine machine{processor_ranges};
  const auto empty_machine_range =
    legate::mapping::ProcessorRange{/*low_id=*/0, /*high_id=*/0, /*per_node_proc_count=*/1};

  ASSERT_EQ(machine.preferred_target(), legate::mapping::TaskTarget::CPU);
  ASSERT_EQ(machine.count(), 2);
  ASSERT_EQ(machine.count(legate::mapping::TaskTarget::GPU), 0);
  ASSERT_EQ(machine.processor_range(), CPU_RANGE);
  ASSERT_EQ(machine.processor_range(legate::mapping::TaskTarget::GPU), empty_machine_range);
  ASSERT_EQ(machine.slice(0, 1),
            (legate::mapping::Machine{
              {{legate::mapping::TaskTarget::CPU, legate::mapping::ProcessorRange{1, 2, 4}}}}));
  ASSERT_FALSE(machine.empty());
  ASSERT_EQ(machine.valid_targets().size(), 1);
  ASSERT_EQ(machine.only(legate::mapping::TaskTarget::CPU),
            (legate::mapping::Machine{{{legate::mapping::TaskTarget::CPU, CPU_RANGE}}}));
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
  const auto valid_targets1 = to_vector(machine1.valid_targets());
  const auto targets1 = std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU,
                                                                 legate::mapping::TaskTarget::OMP,
                                                                 legate::mapping::TaskTarget::GPU};
  ASSERT_TRUE(check_task_target_vec(valid_targets1, targets1));

  const legate::mapping::Machine machine2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::OMP, OMP_RANGE}}};
  const auto valid_targets2 = to_vector(machine2.valid_targets());
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
  const auto machine1       = machine.only(legate::mapping::TaskTarget::CPU);
  const auto valid_targets1 = to_vector(machine1.valid_targets());

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

TEST_F(MachineTest, OnlyEmptyFullMachine)
{
  const legate::mapping::Machine machine{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  ASSERT_EQ(machine.preferred_target(), legate::mapping::TaskTarget::GPU);

  const auto empty = machine.only(legate::Span<const legate::mapping::TaskTarget>{});

  ASSERT_TRUE(empty.empty());
  ASSERT_EQ(empty.preferred_target(), machine.preferred_target());
}

TEST_F(MachineTest, OnlyEmptyEmptyMachine)
{
  constexpr auto EMPTY_RANGE = legate::mapping::ProcessorRange{};

  static_assert(EMPTY_RANGE.empty());

  const legate::mapping::Machine machine{{{legate::mapping::TaskTarget::CPU, EMPTY_RANGE},
                                          {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  ASSERT_EQ(machine.preferred_target(), legate::mapping::TaskTarget::GPU);

  const auto empty = machine.only(legate::Span<const legate::mapping::TaskTarget>{});

  ASSERT_TRUE(empty.empty());
  ASSERT_EQ(empty.preferred_target(), machine.preferred_target());
}

TEST_F(MachineTest, OnlyIf)
{
  const legate::mapping::Machine machine{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                          {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                          {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  auto machine1 = machine.impl()->only_if(
    [](legate::mapping::TaskTarget t) { return t == legate::mapping::TaskTarget::CPU; });
  ASSERT_EQ(to_vector(machine1.valid_targets()),
            std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU});
  ASSERT_EQ(machine1.preferred_target(), legate::mapping::TaskTarget::CPU);
}

TEST_F(MachineTest, Slice)
{
  const legate::mapping::Machine machine1{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE}, {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const legate::mapping::Machine expected{
    {{legate::mapping::TaskTarget::GPU, GPU_RANGE.slice(/*from=*/0, /*to=*/1)}}};

  ASSERT_EQ(machine1.slice(0, 1), expected);

  const auto new_machine1 = machine1.slice(/*from=*/0, /*to=*/2, legate::mapping::TaskTarget::GPU);

  ASSERT_EQ(new_machine1.preferred_target(), legate::mapping::TaskTarget::GPU);
  ASSERT_EQ(new_machine1.processor_range().count(), 2);

  const legate::mapping::Machine machine2{{{legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const auto new_machine2 = machine2.slice(/*from=*/0, /*to=*/2);

  ASSERT_EQ(new_machine2.preferred_target(), legate::mapping::TaskTarget::GPU);
  ASSERT_EQ(new_machine2.processor_range().count(), 2);

  const legate::mapping::Machine machine3{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                           {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                           {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const legate::mapping::Machine expected1{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE},
     {legate::mapping::TaskTarget::OMP, OMP_RANGE},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE.slice(/*from=*/0, /*to=*/1)}}};

  ASSERT_EQ(machine3.slice(0, 1, true), expected1);

  const legate::mapping::Machine expected2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE.slice(/*from=*/1, /*to=*/2)},
     {legate::mapping::TaskTarget::OMP, OMP_RANGE},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  ASSERT_EQ(machine3.slice(1, 2, legate::mapping::TaskTarget::CPU, true), expected2);
}

TEST_F(MachineTest, IndexOperator)
{
  const legate::mapping::Machine machine{{{legate::mapping::TaskTarget::CPU, CPU_RANGE},
                                          {legate::mapping::TaskTarget::OMP, OMP_RANGE},
                                          {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};
  const auto machine1       = machine[legate::mapping::TaskTarget::GPU];
  const auto valid_targets1 = to_vector(machine1.valid_targets());

  ASSERT_EQ(valid_targets1,
            std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::GPU});
  ASSERT_EQ(machine1.count(), 3);
  ASSERT_EQ(machine1.preferred_target(), legate::mapping::TaskTarget::GPU);
  ASSERT_EQ(machine1.processor_range().per_node_count, 3);

  const auto targets  = std::vector<legate::mapping::TaskTarget>{legate::mapping::TaskTarget::CPU,
                                                                 legate::mapping::TaskTarget::OMP};
  const auto machine2 = machine[targets];
  const auto valid_targets2 = to_vector(machine2.valid_targets());

  ASSERT_TRUE(check_task_target_vec(valid_targets2, targets));
  ASSERT_EQ(machine2.preferred_target(), legate::mapping::TaskTarget::OMP);
  ASSERT_EQ(machine2.processor_range().per_node_count, 2);
}

TEST_F(MachineTest, IntersectionOperator)
{
  const legate::mapping::Machine machine1{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE.slice(/*from=*/1, /*to=*/2)},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE}}};

  const legate::mapping::Machine machine2{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE},
     {legate::mapping::TaskTarget::OMP, OMP_RANGE},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE.slice(/*from=*/0, /*to=*/1)}}};

  const legate::mapping::Machine machine3{
    {{legate::mapping::TaskTarget::CPU, CPU_RANGE.slice(/*from=*/1, /*to=*/2)},
     {legate::mapping::TaskTarget::GPU, GPU_RANGE.slice(/*from=*/0, /*to=*/1)}}};
  ASSERT_EQ(machine3, (machine1 & machine2));

  const legate::mapping::Machine machine4{
    {{legate::mapping::TaskTarget::CPU, {/*low_id=*/0, /*high_id=*/0, /*per_node_proc_count=*/1}}}};
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

}  // namespace machine_test
