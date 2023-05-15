/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <valarray>

#include "core/utilities/buffer_builder.h"
#include "legate.h"

namespace unit {

TEST(Machine, ProcessorRange)
{
  // create nonempty
  {
    legate::mapping::ProcessorRange range(1, 3, 1);
    EXPECT_FALSE(range.empty());
    EXPECT_EQ(range.per_node_count, 1);
    EXPECT_EQ(range.low, 1);
    EXPECT_EQ(range.high, 3);
    EXPECT_EQ(range.count(), 2);
    EXPECT_EQ(range.get_node_range(), std::make_pair(uint32_t(1), uint32_t(3)));
  }

  // create empty
  {
    legate::mapping::ProcessorRange range(1, 0, 1);
    EXPECT_TRUE(range.empty());
    EXPECT_EQ(range.per_node_count, 1);
    EXPECT_EQ(range.low, 0);
    EXPECT_EQ(range.high, 0);
    EXPECT_EQ(range.count(), 0);
    EXPECT_THROW(range.get_node_range(), std::runtime_error);
  }

  // check defaults
  {
    legate::mapping::ProcessorRange range;
    EXPECT_TRUE(range.empty());
    EXPECT_EQ(range.per_node_count, 1);
    EXPECT_EQ(range.low, 0);
    EXPECT_EQ(range.high, 0);
    EXPECT_EQ(range.count(), 0);
  }

  // get_node_range
  {
    legate::mapping::ProcessorRange range(0, 7, 2);
    EXPECT_EQ(range.get_node_range(), std::make_pair(uint32_t(0), uint32_t(3)));
  }

  // intersection nonempty
  {
    legate::mapping::ProcessorRange range1(0, 3, 1);
    legate::mapping::ProcessorRange range2(2, 4, 1);
    auto range3 = range1 & range2;
    EXPECT_EQ(range3, legate::mapping::ProcessorRange(2, 3, 1));
  }

  // intersection empty
  {
    legate::mapping::ProcessorRange range1(0, 2, 1);
    legate::mapping::ProcessorRange range2(3, 5, 1);
    auto range3 = range1 & range2;
    EXPECT_EQ(range3, legate::mapping::ProcessorRange(0, 0, 1));
    EXPECT_EQ(range3.count(), 0);
  }

  // empty slice empty range
  {
    legate::mapping::ProcessorRange range(3, 1, 1);
    EXPECT_EQ(range.slice(0, 0).count(), 0);
    EXPECT_EQ(range.slice(4, 6).count(), 0);
  }

  // empty slice nonempty range
  {
    legate::mapping::ProcessorRange range(1, 3, 1);
    EXPECT_EQ(range.slice(0, 0).count(), 0);
    EXPECT_EQ(range.slice(4, 6).count(), 0);
  }

  // nonempty slice nonempty range
  {
    legate::mapping::ProcessorRange range(1, 3, 1);
    EXPECT_EQ(range.slice(1, 3).count(), 1);
    EXPECT_EQ(range.slice(0, 2).count(), 2);
  }
}

TEST(Machine, MachineDesc)
{
  // test empty MachineDesc
  {
    legate::mapping::MachineDesc machine;
    EXPECT_EQ(machine.preferred_target, legate::mapping::TaskTarget::CPU);
    EXPECT_THROW(machine.count(), std::runtime_error);
    EXPECT_THROW(machine.count(legate::mapping::TaskTarget::GPU), std::runtime_error);
    EXPECT_EQ(machine.processor_range(), legate::mapping::ProcessorRange(0, 0, 1));
    EXPECT_EQ(machine.processor_range(legate::mapping::TaskTarget::GPU),
              legate::mapping::ProcessorRange(0, 0, 1));
    EXPECT_EQ(machine.slice(0, 1),
              legate::mapping::MachineDesc(
                {{legate::mapping::TaskTarget::CPU, legate::mapping::ProcessorRange()}}));
  }

  legate::mapping::ProcessorRange cpu_range(1, 3, 4);
  legate::mapping::ProcessorRange omp_range(0, 3, 2);
  legate::mapping::ProcessorRange gpu_range(3, 6, 3);
  legate::mapping::ProcessorRange empty_range;

  // test equal
  {
    legate::mapping::MachineDesc machine1({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range}});
    legate::mapping::MachineDesc machine2({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range}});
    EXPECT_EQ(machine1, machine2);

    legate::mapping::MachineDesc machine3;
    EXPECT_NE(machine1, machine3);
  }

  // test preferred_target
  {
    legate::mapping::MachineDesc machine1({{legate::mapping::TaskTarget::CPU, cpu_range}});
    EXPECT_EQ(machine1.preferred_target, legate::mapping::TaskTarget::CPU);

    legate::mapping::MachineDesc machine2({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range}});
    EXPECT_EQ(machine2.preferred_target, legate::mapping::TaskTarget::OMP);

    legate::mapping::MachineDesc machine3({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range},
                                           {legate::mapping::TaskTarget::GPU, gpu_range}});
    EXPECT_EQ(machine3.preferred_target, legate::mapping::TaskTarget::GPU);
  }

  // test processor_range
  {
    legate::mapping::MachineDesc machine1({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range},
                                           {legate::mapping::TaskTarget::GPU, gpu_range}});
    EXPECT_EQ(machine1.processor_range(), gpu_range);
    EXPECT_EQ(machine1.processor_range(legate::mapping::TaskTarget::CPU), cpu_range);
    EXPECT_EQ(machine1.processor_range(legate::mapping::TaskTarget::OMP), omp_range);
    EXPECT_EQ(machine1.processor_range(legate::mapping::TaskTarget::GPU), gpu_range);

    legate::mapping::MachineDesc machine2({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range}});
    EXPECT_EQ(machine2.processor_range(), omp_range);
    EXPECT_EQ(machine2.processor_range(legate::mapping::TaskTarget::CPU), cpu_range);
    EXPECT_EQ(machine2.processor_range(legate::mapping::TaskTarget::GPU),
              legate::mapping::ProcessorRange());
  }

  // test valid_targets
  {
    legate::mapping::MachineDesc machine1({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range},
                                           {legate::mapping::TaskTarget::GPU, gpu_range}});
    auto valid_targets1 = machine1.valid_targets();
    EXPECT_EQ(valid_targets1.size(), 3);

    legate::mapping::MachineDesc machine2({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range}});
    auto valid_targets2 = machine2.valid_targets();
    EXPECT_EQ(valid_targets2.size(), 2);
  }

  // test valid_targets_except
  {
    legate::mapping::MachineDesc machine({{legate::mapping::TaskTarget::CPU, cpu_range},
                                          {legate::mapping::TaskTarget::OMP, omp_range},
                                          {legate::mapping::TaskTarget::GPU, gpu_range}});

    std::set<legate::mapping::TaskTarget> exclude_targets;
    auto valid_targets1 = machine.valid_targets_except(exclude_targets);
    EXPECT_EQ(valid_targets1.size(), 3);

    exclude_targets.insert(legate::mapping::TaskTarget::CPU);
    auto valid_targets2 = machine.valid_targets_except(exclude_targets);
    EXPECT_EQ(valid_targets2.size(), 2);

    exclude_targets.insert(legate::mapping::TaskTarget::OMP);
    auto valid_targets3 = machine.valid_targets_except(exclude_targets);
    EXPECT_EQ(valid_targets3.size(), 1);

    exclude_targets.insert(legate::mapping::TaskTarget::GPU);
    auto valid_targets4 = machine.valid_targets_except(exclude_targets);
    EXPECT_EQ(valid_targets4.size(), 0);
  }

  // test count
  {
    legate::mapping::MachineDesc machine({{legate::mapping::TaskTarget::CPU, cpu_range},
                                          {legate::mapping::TaskTarget::OMP, omp_range},
                                          {legate::mapping::TaskTarget::GPU, gpu_range}});
    EXPECT_EQ(machine.count(), 3);
    EXPECT_EQ(machine.count(legate::mapping::TaskTarget::CPU), 2);
    EXPECT_EQ(machine.count(legate::mapping::TaskTarget::OMP), 3);
  }

  // test_pack
  {
    legate::BufferBuilder buf;
    legate::mapping::MachineDesc machine({{legate::mapping::TaskTarget::CPU, cpu_range},
                                          {legate::mapping::TaskTarget::GPU, gpu_range}});
    machine.pack(buf);
    auto legion_buffer = buf.to_legion_buffer();
    legate::BaseDeserializer<legate::mapping::MapperDataDeserializer> dez(legion_buffer.get_ptr(),
                                                                          legion_buffer.get_size());
    auto machine_unpack = dez.unpack<legate::mapping::MachineDesc>();
    EXPECT_EQ(machine_unpack, machine);
  }

  // test only
  {
    legate::mapping::MachineDesc machine({{legate::mapping::TaskTarget::CPU, cpu_range},
                                          {legate::mapping::TaskTarget::GPU, gpu_range}});

    auto machine1       = machine.only(legate::mapping::TaskTarget::CPU);
    auto valid_targets1 = machine1.valid_targets();
    EXPECT_EQ(valid_targets1.size(), 1);
    EXPECT_EQ(machine1.count(), 2);
    EXPECT_EQ(machine1.preferred_target, legate::mapping::TaskTarget::CPU);
    EXPECT_EQ(machine1.processor_range().per_node_count, 4);

    legate::mapping::MachineDesc machine2({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::OMP, omp_range},
                                           {legate::mapping::TaskTarget::GPU, gpu_range}});
    auto machine3 =
      machine2.only({legate::mapping::TaskTarget::GPU, legate::mapping::TaskTarget::CPU});
    EXPECT_EQ(machine3, machine);
  }

  // test slice
  {
    legate::mapping::MachineDesc machine1({{legate::mapping::TaskTarget::CPU, cpu_range},
                                           {legate::mapping::TaskTarget::GPU, gpu_range}});
    EXPECT_THROW(machine1.slice(0, 1), std::runtime_error);

    auto new_machine1 = machine1.slice(0, 2, legate::mapping::TaskTarget::GPU);
    EXPECT_EQ(new_machine1.preferred_target, legate::mapping::TaskTarget::GPU);
    EXPECT_EQ(new_machine1.processor_range().count(), 2);

    legate::mapping::MachineDesc machine2({{legate::mapping::TaskTarget::GPU, gpu_range}});
    auto new_machine2 = machine2.slice(0, 2);
    EXPECT_EQ(new_machine2.preferred_target, legate::mapping::TaskTarget::GPU);
    EXPECT_EQ(new_machine2.processor_range().count(), 2);
  }

  // test operator[]
  {
    legate::mapping::MachineDesc machine({{legate::mapping::TaskTarget::CPU, cpu_range},
                                          {legate::mapping::TaskTarget::GPU, gpu_range}});

    auto machine1       = machine[legate::mapping::TaskTarget::GPU];
    auto valid_targets1 = machine1.valid_targets();
    EXPECT_EQ(valid_targets1.size(), 1);
    EXPECT_EQ(machine1.count(), 3);
    EXPECT_EQ(machine1.preferred_target, legate::mapping::TaskTarget::GPU);
    EXPECT_EQ(machine1.processor_range().per_node_count, 3);
  }

  // test intersection
  {
    legate::mapping::MachineDesc machine1(
      {{legate::mapping::TaskTarget::CPU, cpu_range.slice(1, 2)},
       {legate::mapping::TaskTarget::GPU, gpu_range}});

    legate::mapping::MachineDesc machine2(
      {{legate::mapping::TaskTarget::CPU, cpu_range},
       {legate::mapping::TaskTarget::OMP, omp_range},
       {legate::mapping::TaskTarget::GPU, gpu_range.slice(0, 1)}});

    legate::mapping::MachineDesc machine3(
      {{legate::mapping::TaskTarget::CPU, cpu_range.slice(1, 2)},
       {legate::mapping::TaskTarget::GPU, gpu_range.slice(0, 1)}});

    EXPECT_EQ(machine3, (machine1 & machine2));
    EXPECT_EQ(machine2.only(legate::mapping::TaskTarget::OMP) & machine1,
              legate::mapping::MachineDesc());
  }
}

}  // namespace unit
