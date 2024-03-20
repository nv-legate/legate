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

#include "core/mapping/detail/machine.h"

#include "core/mapping/machine.h"
#include "core/utilities/deserializer.h"
#include "core/utilities/detail/buffer_builder.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <valarray>

namespace unit {

using Machine = DefaultFixture;

// NOLINTBEGIN(readability-magic-numbers)

TEST_F(Machine, ProcessorRange)
{
  // create nonempty
  {
    const legate::mapping::ProcessorRange range{1, 3, 1};

    EXPECT_FALSE(range.empty());
    EXPECT_EQ(range.per_node_count, 1);
    EXPECT_EQ(range.low, 1);
    EXPECT_EQ(range.high, 3);
    EXPECT_EQ(range.count(), 2);
    EXPECT_EQ(range.get_node_range(), legate::mapping::NodeRange({1, 3}));
  }

  // create empty
  {
    const legate::mapping::ProcessorRange range{1, 0, 1};

    EXPECT_TRUE(range.empty());
    EXPECT_EQ(range.per_node_count, 1);
    EXPECT_EQ(range.low, 0);
    EXPECT_EQ(range.high, 0);
    EXPECT_EQ(range.count(), 0);
    EXPECT_THROW(static_cast<void>(range.get_node_range()), std::invalid_argument);
  }

  // create another empty
  {
    const legate::mapping::ProcessorRange range{3, 3, 0};

    EXPECT_TRUE(range.empty());
    EXPECT_EQ(range.per_node_count, 1);
    EXPECT_EQ(range.low, 0);
    EXPECT_EQ(range.high, 0);
    EXPECT_EQ(range.count(), 0);
    EXPECT_THROW(static_cast<void>(range.get_node_range()), std::invalid_argument);
  }

  // check defaults
  {
    const legate::mapping::ProcessorRange range;

    EXPECT_TRUE(range.empty());
    EXPECT_EQ(range.per_node_count, 1);
    EXPECT_EQ(range.low, 0);
    EXPECT_EQ(range.high, 0);
    EXPECT_EQ(range.count(), 0);
  }

  // test equal and comparison
  {
    const legate::mapping::ProcessorRange range1{2, 6, 2};
    const legate::mapping::ProcessorRange range2{2, 6, 2};

    EXPECT_EQ(range1, range2);

    const legate::mapping::ProcessorRange range3{1, 6, 2};

    EXPECT_NE(range1, range3);
    EXPECT_TRUE(range3 < range1);

    const legate::mapping::ProcessorRange range4{2, 5, 2};

    EXPECT_TRUE(range4 < range1);

    const legate::mapping::ProcessorRange range5{2, 6, 1};

    EXPECT_TRUE(range5 < range1);
  }

  // get_node_range
  {
    const legate::mapping::ProcessorRange range{0, 7, 2};

    EXPECT_EQ(range.get_node_range(), legate::mapping::NodeRange({0, 4}));
  }

  // intersection nonempty
  {
    const legate::mapping::ProcessorRange range1{0, 3, 1};
    const legate::mapping::ProcessorRange range2{2, 4, 1};
    const auto range3 = range1 & range2;

    EXPECT_EQ(range3, (legate::mapping::ProcessorRange{2, 3, 1}));
  }

  // intersection empty
  {
    const legate::mapping::ProcessorRange range1{0, 2, 1};
    const legate::mapping::ProcessorRange range2{3, 5, 1};
    const auto range3 = range1 & range2;

    EXPECT_EQ(range3, (legate::mapping::ProcessorRange{0, 0, 1}));
    EXPECT_EQ(range3.count(), 0);
  }

  // empty slice empty range
  {
    const legate::mapping::ProcessorRange range{3, 1, 1};

    EXPECT_EQ(range.slice(0, 0).count(), 0);
    EXPECT_EQ(range.slice(4, 6).count(), 0);
  }

  // empty slice nonempty range
  {
    const legate::mapping::ProcessorRange range{1, 3, 1};

    EXPECT_EQ(range.slice(0, 0).count(), 0);
    EXPECT_EQ(range.slice(4, 6).count(), 0);
    EXPECT_EQ(range.slice(1, 0).count(), 0);
  }

  // nonempty slice nonempty range
  {
    const legate::mapping::ProcessorRange range{1, 3, 1};

    EXPECT_EQ(range.slice(1, 3).count(), 1);
    EXPECT_EQ(range.slice(0, 2).count(), 2);
  }
}

TEST_F(Machine, MachineDesc)
{
  // test empty MachineDesc
  {
    const legate::mapping::detail::Machine machine;

    EXPECT_EQ(machine.preferred_target, legate::mapping::TaskTarget::CPU);
    EXPECT_EQ(machine.count(), 0);
    EXPECT_EQ(machine.count(legate::mapping::TaskTarget::GPU), 0);
    EXPECT_EQ(machine.processor_range(), legate::mapping::ProcessorRange(0, 0, 1));
    EXPECT_EQ(machine.processor_range(legate::mapping::TaskTarget::GPU),
              legate::mapping::ProcessorRange(0, 0, 1));
    EXPECT_EQ(machine.slice(0, 1),
              legate::mapping::detail::Machine(
                {{legate::mapping::TaskTarget::CPU, legate::mapping::ProcessorRange()}}));
    EXPECT_TRUE(machine.empty());
    EXPECT_EQ(machine.valid_targets().size(), 0);
    EXPECT_EQ(machine.only(legate::mapping::TaskTarget::CPU),
              legate::mapping::detail::Machine(
                {{legate::mapping::TaskTarget::CPU, legate::mapping::ProcessorRange()}}));

    const std::map<legate::mapping::TaskTarget, legate::mapping::ProcessorRange> processor_ranges{};

    EXPECT_EQ(machine.processor_ranges, processor_ranges);
  }

  legate::mapping::ProcessorRange cpu_range{1, 3, 4};
  legate::mapping::ProcessorRange omp_range{0, 3, 2};
  legate::mapping::ProcessorRange gpu_range{3, 6, 3};

  // test equal
  {
    const legate::mapping::detail::Machine machine1{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range}}};
    const legate::mapping::detail::Machine machine2{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range}}};

    EXPECT_EQ(machine1, machine2);

    const legate::mapping::detail::Machine machine3;

    EXPECT_NE(machine1, machine3);
  }

  // test preferred_target
  {
    const legate::mapping::detail::Machine machine1{
      {{legate::mapping::TaskTarget::CPU, cpu_range}}};

    EXPECT_EQ(machine1.preferred_target, legate::mapping::TaskTarget::CPU);

    const legate::mapping::detail::Machine machine2{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range}}};

    EXPECT_EQ(machine2.preferred_target, legate::mapping::TaskTarget::OMP);

    const legate::mapping::detail::Machine machine3{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range}}};

    EXPECT_EQ(machine3.preferred_target, legate::mapping::TaskTarget::GPU);
  }

  // test processor_range
  {
    const legate::mapping::detail::Machine machine1{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range}}};

    EXPECT_EQ(machine1.processor_range(), gpu_range);
    EXPECT_EQ(machine1.processor_range(legate::mapping::TaskTarget::CPU), cpu_range);
    EXPECT_EQ(machine1.processor_range(legate::mapping::TaskTarget::OMP), omp_range);
    EXPECT_EQ(machine1.processor_range(legate::mapping::TaskTarget::GPU), gpu_range);

    const legate::mapping::detail::Machine machine2{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range}}};

    EXPECT_EQ(machine2.processor_range(), omp_range);
    EXPECT_EQ(machine2.processor_range(legate::mapping::TaskTarget::CPU), cpu_range);
    EXPECT_EQ(machine2.processor_range(legate::mapping::TaskTarget::GPU),
              legate::mapping::ProcessorRange{});
  }

  // test valid_targets
  {
    const legate::mapping::detail::Machine machine1{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range}}};
    const auto valid_targets1 = machine1.valid_targets();

    EXPECT_EQ(valid_targets1.size(), 3);

    const legate::mapping::detail::Machine machine2{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range}}};
    const auto valid_targets2 = machine2.valid_targets();

    EXPECT_EQ(valid_targets2.size(), 2);
  }

  // test valid_targets_except
  {
    const legate::mapping::detail::Machine machine{{{legate::mapping::TaskTarget::CPU, cpu_range},
                                                    {legate::mapping::TaskTarget::OMP, omp_range},
                                                    {legate::mapping::TaskTarget::GPU, gpu_range}}};
    std::set<legate::mapping::TaskTarget> exclude_targets;
    const auto valid_targets1 = machine.valid_targets_except(exclude_targets);

    EXPECT_EQ(valid_targets1.size(), 3);

    exclude_targets.insert(legate::mapping::TaskTarget::CPU);
    const auto valid_targets2 = machine.valid_targets_except(exclude_targets);

    EXPECT_EQ(valid_targets2.size(), 2);

    exclude_targets.insert(legate::mapping::TaskTarget::OMP);
    const auto valid_targets3 = machine.valid_targets_except(exclude_targets);
    EXPECT_EQ(valid_targets3.size(), 1);

    exclude_targets.insert(legate::mapping::TaskTarget::GPU);
    const auto valid_targets4 = machine.valid_targets_except(exclude_targets);
    EXPECT_EQ(valid_targets4.size(), 0);
  }

  // test count
  {
    const legate::mapping::detail::Machine machine{{{legate::mapping::TaskTarget::CPU, cpu_range},
                                                    {legate::mapping::TaskTarget::OMP, omp_range},
                                                    {legate::mapping::TaskTarget::GPU, gpu_range}}};

    EXPECT_EQ(machine.count(), 3);
    EXPECT_EQ(machine.count(legate::mapping::TaskTarget::CPU), 2);
    EXPECT_EQ(machine.count(legate::mapping::TaskTarget::OMP), 3);
  }

  // test_pack
  {
    legate::detail::BufferBuilder buf;
    const legate::mapping::detail::Machine machine{{{legate::mapping::TaskTarget::CPU, cpu_range},
                                                    {legate::mapping::TaskTarget::GPU, gpu_range}}};
    machine.pack(buf);
    auto legion_buffer = buf.to_legion_buffer();
    legate::BaseDeserializer<legate::mapping::MapperDataDeserializer> dez{legion_buffer.get_ptr(),
                                                                          legion_buffer.get_size()};
    auto machine_unpack = dez.unpack<legate::mapping::detail::Machine>();

    EXPECT_EQ(machine_unpack, machine);
  }

  // test only
  {
    const legate::mapping::detail::Machine machine{{{legate::mapping::TaskTarget::CPU, cpu_range},
                                                    {legate::mapping::TaskTarget::GPU, gpu_range}}};
    const auto machine1       = machine.only(legate::mapping::TaskTarget::CPU);
    const auto valid_targets1 = machine1.valid_targets();

    EXPECT_EQ(valid_targets1.size(), 1);
    EXPECT_EQ(machine1.count(), 2);
    EXPECT_EQ(machine1.preferred_target, legate::mapping::TaskTarget::CPU);
    EXPECT_EQ(machine1.processor_range().per_node_count, 4);

    const legate::mapping::detail::Machine machine2{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range}}};
    const auto machine3 =
      machine2.only({legate::mapping::TaskTarget::GPU, legate::mapping::TaskTarget::CPU});

    EXPECT_EQ(machine3, machine);
  }

  // test slice
  {
    const legate::mapping::detail::Machine machine1{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::GPU, gpu_range}}};
    const legate::mapping::detail::Machine expected{
      {{legate::mapping::TaskTarget::GPU, gpu_range.slice(0, 1)}}};

    EXPECT_EQ(machine1.slice(0, 1), expected);

    const auto new_machine1 = machine1.slice(0, 2, legate::mapping::TaskTarget::GPU);

    EXPECT_EQ(new_machine1.preferred_target, legate::mapping::TaskTarget::GPU);
    EXPECT_EQ(new_machine1.processor_range().count(), 2);

    const legate::mapping::detail::Machine machine2{
      {{legate::mapping::TaskTarget::GPU, gpu_range}}};
    const auto new_machine2 = machine2.slice(0, 2);

    EXPECT_EQ(new_machine2.preferred_target, legate::mapping::TaskTarget::GPU);
    EXPECT_EQ(new_machine2.processor_range().count(), 2);

    const legate::mapping::detail::Machine machine3{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range}}};
    const legate::mapping::detail::Machine expected1{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range.slice(0, 1)}}};

    EXPECT_EQ(machine3.slice(0, 1, true), expected1);

    const legate::mapping::detail::Machine expected2{
      {{legate::mapping::TaskTarget::CPU, cpu_range.slice(1, 2)},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range}}};

    EXPECT_EQ(machine3.slice(1, 2, legate::mapping::TaskTarget::CPU, true), expected2);
  }

  // test operator[]
  {
    const legate::mapping::detail::Machine machine{{{legate::mapping::TaskTarget::CPU, cpu_range},
                                                    {legate::mapping::TaskTarget::OMP, omp_range},
                                                    {legate::mapping::TaskTarget::GPU, gpu_range}}};
    const auto machine1       = machine[legate::mapping::TaskTarget::GPU];
    const auto valid_targets1 = machine1.valid_targets();

    EXPECT_EQ(valid_targets1.size(), 1);
    EXPECT_EQ(machine1.count(), 3);
    EXPECT_EQ(machine1.preferred_target, legate::mapping::TaskTarget::GPU);
    EXPECT_EQ(machine1.processor_range().per_node_count, 3);

    const auto machine2 =
      machine[{legate::mapping::TaskTarget::CPU, legate::mapping::TaskTarget::OMP}];
    const auto valid_targets2 = machine2.valid_targets();

    EXPECT_EQ(valid_targets2.size(), 2);
    EXPECT_EQ(machine2.preferred_target, legate::mapping::TaskTarget::OMP);
    EXPECT_EQ(machine2.processor_range().per_node_count, 2);
  }

  // test intersection
  {
    const legate::mapping::detail::Machine machine1{
      {{legate::mapping::TaskTarget::CPU, cpu_range.slice(1, 2)},
       {legate::mapping::TaskTarget::GPU, gpu_range}}};

    const legate::mapping::detail::Machine machine2{
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range.slice(0, 1)}}};

    const legate::mapping::detail::Machine machine3{
      {{legate::mapping::TaskTarget::CPU, cpu_range.slice(1, 2)},
       {legate::mapping::TaskTarget::GPU, gpu_range.slice(0, 1)}}};

    EXPECT_EQ(machine3, (machine1 & machine2));
    EXPECT_EQ(machine2.only(legate::mapping::TaskTarget::OMP) & machine1,
              legate::mapping::detail::Machine{});
  }
}

// NOLINTEND(readability-magic-numbers)

}  // namespace unit
