/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/base_mapper.h>

#include <legate.h>

#include <legate/data/detail/logical_store_partition.h>
#include <legate/data/detail/storage_partition.h>
#include <legate/mapping/detail/operation.h>
#include <legate/utilities/detail/buffer_builder.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <utilities/utilities.h>

namespace base_mapper_unit {

namespace {

using BaseMapperTest = DefaultFixture;

class TestPartition final : public Legion::Partition {
 public:
  [[nodiscard]] Legion::UniqueID get_unique_id() const override
  {
    ADD_FAILURE() << "TestPartition::get_unique_id() should not be called";
    return 0;
  }

  [[nodiscard]] std::uint64_t get_context_index() const override
  {
    ADD_FAILURE() << "TestPartition::get_context_index() should not be called";
    return 0;
  }

  [[nodiscard]] int get_depth() const override
  {
    ADD_FAILURE() << "TestPartition::get_depth() should not be called";
    return 0;
  }

  [[nodiscard]] const Legion::Task* get_parent_task() const override
  {
    ADD_FAILURE() << "TestPartition::get_parent_task() should not be called";
    return nullptr;
  }

  [[nodiscard]] const std::string_view& get_provenance_string(bool) const override
  {
    ADD_FAILURE() << "TestPartition::get_provenance_string() should not be called";
    static constexpr std::string_view provenance{};
    return provenance;
  }

  [[nodiscard]] PartitionKind get_partition_kind() const override
  {
    ADD_FAILURE() << "TestPartition::get_partition_kind() should not be called";
    return BY_FIELD;
  }
};

class TestMappable final : public Legion::Mappable {
 public:
  [[nodiscard]] Legion::UniqueID get_unique_id() const override
  {
    ADD_FAILURE() << "TestMappable::get_unique_id() should not be called";
    return 0;
  }

  [[nodiscard]] std::uint64_t get_context_index() const override
  {
    ADD_FAILURE() << "TestMappable::get_context_index() should not be called";
    return 0;
  }

  [[nodiscard]] int get_depth() const override
  {
    ADD_FAILURE() << "TestMappable::get_depth() should not be called";
    return 0;
  }

  [[nodiscard]] const Legion::Task* get_parent_task() const override
  {
    ADD_FAILURE() << "TestMappable::get_parent_task() should not be called";
    return nullptr;
  }

  [[nodiscard]] const std::string_view& get_provenance_string(bool) const override
  {
    ADD_FAILURE() << "TestMappable::get_provenance_string() should not be called";
    static constexpr std::string_view provenance{};
    return provenance;
  }

  [[nodiscard]] Legion::MappableType get_mappable_type() const override
  {
    ADD_FAILURE() << "TestMappable::get_mappable_type() should not be called";
    return LEGION_TASK_MAPPABLE;
  }
};

Legion::LogicalPartition create_test_partition()
{
  auto runtime         = legate::Runtime::get_runtime();
  auto store           = runtime->create_store(legate::Shape{4}, legate::int32());
  auto store_partition = store.partition_by_tiling({2});

  return store_partition.impl()->storage_partition()->get_legion_partition();
}

}  // namespace

TEST_F(BaseMapperTest, SelectPartitionProjectionUsesOpenCompletePartition)
{
  legate::mapping::detail::BaseMapper mapper;
  const TestPartition partition;
  auto input_partition = create_test_partition();
  Legion::Mapping::Mapper::SelectPartitionProjectionInput input;
  Legion::Mapping::Mapper::SelectPartitionProjectionOutput output;

  input.open_complete_partitions.push_back(input_partition);

  mapper.select_partition_projection(/*ctx=*/nullptr, partition, input, output);

  ASSERT_EQ(output.chosen_partition, input_partition);
}

TEST_F(BaseMapperTest, SelectPartitionProjectionUsesNoPartForEmptyInput)
{
  legate::mapping::detail::BaseMapper mapper;
  const TestPartition partition;
  const Legion::Mapping::Mapper::SelectPartitionProjectionInput input;
  Legion::Mapping::Mapper::SelectPartitionProjectionOutput output;

  mapper.select_partition_projection(/*ctx=*/nullptr, partition, input, output);

  ASSERT_EQ(output.chosen_partition, Legion::LogicalPartition::NO_PART);
}

TEST_F(BaseMapperTest, ProcessorAccessorsMatchLocalMachine)
{
  const legate::mapping::detail::BaseMapper mapper;
  const legate::mapping::detail::LocalMachine local_machine;

  ASSERT_EQ(mapper.cpus().size(), local_machine.cpus().size());
  ASSERT_EQ(mapper.gpus().size(), local_machine.gpus().size());
  ASSERT_EQ(mapper.omps().size(), local_machine.omps().size());
  ASSERT_EQ(mapper.total_nodes(), local_machine.total_nodes);
}

TEST_F(BaseMapperTest, MappableShardingIdAccessor)
{
  legate::detail::BufferBuilder buffer;
  const std::optional<legate::detail::StreamingGeneration> streaming_generation{};
  const legate::mapping::detail::Machine machine{};
  constexpr std::uint32_t expected_key_projection_id = 0;
  constexpr std::uint32_t expected_sharding_id       = 123;
  constexpr std::int32_t priority                    = 7;

  // Keep this prefix in sync with Mappable::Mappable(private_tag, MapperDataDeserializer).
  buffer.pack(streaming_generation);
  machine.pack(buffer);
  buffer.pack(expected_key_projection_id);
  buffer.pack(expected_sharding_id);
  buffer.pack(priority);

  const auto legion_buffer = buffer.to_legion_buffer();
  TestMappable legion_mappable;

  legion_mappable.mapper_data      = legion_buffer.get_ptr();
  legion_mappable.mapper_data_size = legion_buffer.get_size();

  const auto mappable = legate::mapping::detail::Mappable{legion_mappable};

  ASSERT_EQ(mappable.sharding_id(), expected_sharding_id);
}

}  // namespace base_mapper_unit
