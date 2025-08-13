/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/mapping/detail/mapping.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace common_api_unit {

namespace {

using MappingCommonTest = DefaultFixture;

// Test classes for parameterized tests
class ProcessorToTaskTargetTest
  : public MappingCommonTest,
    public ::testing::WithParamInterface<
      std::tuple<Legion::Processor::Kind, legate::mapping::TaskTarget>> {};

class StoreTargetToTaskTargetTest
  : public MappingCommonTest,
    public ::testing::WithParamInterface<
      std::tuple<legate::mapping::StoreTarget, legate::mapping::TaskTarget>> {};

class MemoryToStoreTargetTest : public MappingCommonTest,
                                public ::testing::WithParamInterface<
                                  std::tuple<Legion::Memory::Kind, legate::mapping::StoreTarget>> {
};

class TaskTargetToProcessorTest
  : public MappingCommonTest,
    public ::testing::WithParamInterface<
      std::tuple<legate::mapping::TaskTarget, Legion::Processor::Kind>> {};

class VariantCodeToProcessorTest
  : public MappingCommonTest,
    public ::testing::WithParamInterface<std::tuple<legate::VariantCode, Legion::Processor::Kind>> {
};

class StoreTargetToMemoryTest : public MappingCommonTest,
                                public ::testing::WithParamInterface<
                                  std::tuple<legate::mapping::StoreTarget, Legion::Memory::Kind>> {
};

class TaskTargetToVariantCodeTest
  : public MappingCommonTest,
    public ::testing::WithParamInterface<
      std::tuple<legate::mapping::TaskTarget, legate::VariantCode>> {};

class ProcessorToVariantCodeTest
  : public MappingCommonTest,
    public ::testing::WithParamInterface<std::tuple<Legion::Processor::Kind, legate::VariantCode>> {
};

// Test data for Processor::Kind to TaskTarget conversion
INSTANTIATE_TEST_SUITE_P(
  MappingCommonTest,
  ProcessorToTaskTargetTest,
  ::testing::Values(
    std::make_tuple(Legion::Processor::Kind::TOC_PROC, legate::mapping::TaskTarget::GPU),
    std::make_tuple(Legion::Processor::Kind::OMP_PROC, legate::mapping::TaskTarget::OMP),
    std::make_tuple(Legion::Processor::Kind::LOC_PROC, legate::mapping::TaskTarget::CPU),
    std::make_tuple(Legion::Processor::Kind::PY_PROC, legate::mapping::TaskTarget::CPU)));

// Test data for StoreTarget to TaskTarget matching
INSTANTIATE_TEST_SUITE_P(
  MappingCommonTest,
  StoreTargetToTaskTargetTest,
  ::testing::Values(
    std::make_tuple(legate::mapping::StoreTarget::FBMEM, legate::mapping::TaskTarget::GPU),
    std::make_tuple(legate::mapping::StoreTarget::ZCMEM, legate::mapping::TaskTarget::GPU),
    std::make_tuple(legate::mapping::StoreTarget::SOCKETMEM, legate::mapping::TaskTarget::OMP),
    std::make_tuple(legate::mapping::StoreTarget::SYSMEM, legate::mapping::TaskTarget::CPU)));

// Test data for Memory::Kind to StoreTarget conversion
INSTANTIATE_TEST_SUITE_P(
  MappingCommonTest,
  MemoryToStoreTargetTest,
  ::testing::Values(
    std::make_tuple(Legion::Memory::Kind::SYSTEM_MEM, legate::mapping::StoreTarget::SYSMEM),
    std::make_tuple(Legion::Memory::Kind::GPU_FB_MEM, legate::mapping::StoreTarget::FBMEM),
    std::make_tuple(Legion::Memory::Kind::Z_COPY_MEM, legate::mapping::StoreTarget::ZCMEM),
    std::make_tuple(Legion::Memory::Kind::SOCKET_MEM, legate::mapping::StoreTarget::SOCKETMEM)));

// Test data for TaskTarget to Processor::Kind conversion
INSTANTIATE_TEST_SUITE_P(MappingCommonTest,
                         TaskTargetToProcessorTest,
                         ::testing::Values(std::make_tuple(legate::mapping::TaskTarget::GPU,
                                                           Legion::Processor::Kind::TOC_PROC),
                                           std::make_tuple(legate::mapping::TaskTarget::OMP,
                                                           Legion::Processor::Kind::OMP_PROC),
                                           std::make_tuple(legate::mapping::TaskTarget::CPU,
                                                           Legion::Processor::Kind::LOC_PROC)));

// Test data for VariantCode to Processor::Kind conversion
INSTANTIATE_TEST_SUITE_P(
  MappingCommonTest,
  VariantCodeToProcessorTest,
  ::testing::Values(std::make_tuple(legate::VariantCode::CPU, Legion::Processor::Kind::LOC_PROC),
                    std::make_tuple(legate::VariantCode::GPU, Legion::Processor::Kind::TOC_PROC),
                    std::make_tuple(legate::VariantCode::OMP, Legion::Processor::Kind::OMP_PROC)));

// Test data for StoreTarget to Memory::Kind conversion
INSTANTIATE_TEST_SUITE_P(
  MappingCommonTest,
  StoreTargetToMemoryTest,
  ::testing::Values(
    std::make_tuple(legate::mapping::StoreTarget::SYSMEM, Legion::Memory::Kind::SYSTEM_MEM),
    std::make_tuple(legate::mapping::StoreTarget::FBMEM, Legion::Memory::Kind::GPU_FB_MEM),
    std::make_tuple(legate::mapping::StoreTarget::ZCMEM, Legion::Memory::Kind::Z_COPY_MEM),
    std::make_tuple(legate::mapping::StoreTarget::SOCKETMEM, Legion::Memory::Kind::SOCKET_MEM)));

// Test data for TaskTarget to VariantCode conversion
INSTANTIATE_TEST_SUITE_P(
  MappingCommonTest,
  TaskTargetToVariantCodeTest,
  ::testing::Values(std::make_tuple(legate::mapping::TaskTarget::GPU, legate::VariantCode::GPU),
                    std::make_tuple(legate::mapping::TaskTarget::OMP, legate::VariantCode::OMP),
                    std::make_tuple(legate::mapping::TaskTarget::CPU, legate::VariantCode::CPU)));

// Test data for Processor::Kind to VariantCode conversion (via TaskTarget)
INSTANTIATE_TEST_SUITE_P(
  MappingCommonTest,
  ProcessorToVariantCodeTest,
  ::testing::Values(std::make_tuple(Legion::Processor::Kind::TOC_PROC, legate::VariantCode::GPU),
                    std::make_tuple(Legion::Processor::Kind::OMP_PROC, legate::VariantCode::OMP),
                    std::make_tuple(Legion::Processor::Kind::LOC_PROC, legate::VariantCode::CPU),
                    std::make_tuple(Legion::Processor::Kind::PY_PROC, legate::VariantCode::CPU)));

}  // namespace

// Test to_target(Processor::Kind) function
TEST_P(ProcessorToTaskTargetTest, ProcessorKindToTaskTarget)
{
  const auto [processor_kind, expected_task_target] = GetParam();
  const auto actual_task_target = legate::mapping::detail::to_target(processor_kind);
  ASSERT_EQ(actual_task_target, expected_task_target);
}

// Test get_matching_task_target(StoreTarget) function
TEST_P(StoreTargetToTaskTargetTest, StoreTargetToMatchingTaskTarget)
{
  const auto [store_target, expected_task_target] = GetParam();
  const auto actual_task_target = legate::mapping::detail::get_matching_task_target(store_target);
  ASSERT_EQ(actual_task_target, expected_task_target);
}

// Test to_target(Memory::Kind) function
TEST_P(MemoryToStoreTargetTest, MemoryKindToStoreTarget)
{
  const auto [memory_kind, expected_store_target] = GetParam();
  const auto actual_store_target                  = legate::mapping::detail::to_target(memory_kind);
  ASSERT_EQ(actual_store_target, expected_store_target);
}

// Test to_kind(TaskTarget) function
TEST_P(TaskTargetToProcessorTest, TaskTargetToProcessorKind)
{
  const auto [task_target, expected_processor_kind] = GetParam();
  const auto actual_processor_kind                  = legate::mapping::detail::to_kind(task_target);
  ASSERT_EQ(actual_processor_kind, expected_processor_kind);
}

// Test to_kind(VariantCode) function
TEST_P(VariantCodeToProcessorTest, VariantCodeToProcessorKind)
{
  const auto [variant_code, expected_processor_kind] = GetParam();
  const auto actual_processor_kind = legate::mapping::detail::to_kind(variant_code);
  ASSERT_EQ(actual_processor_kind, expected_processor_kind);
}

// Test to_kind(StoreTarget) function
TEST_P(StoreTargetToMemoryTest, StoreTargetToMemoryKind)
{
  const auto [store_target, expected_memory_kind] = GetParam();
  const auto actual_memory_kind                   = legate::mapping::detail::to_kind(store_target);
  ASSERT_EQ(actual_memory_kind, expected_memory_kind);
}

// Test to_variant_code(TaskTarget) function
TEST_P(TaskTargetToVariantCodeTest, TaskTargetToVariantCode)
{
  const auto [task_target, expected_variant_code] = GetParam();
  const auto actual_variant_code = legate::mapping::detail::to_variant_code(task_target);
  ASSERT_EQ(actual_variant_code, expected_variant_code);
}

// Test to_variant_code(Processor::Kind) function (composite function)
TEST_P(ProcessorToVariantCodeTest, ProcessorKindToVariantCode)
{
  const auto [processor_kind, expected_variant_code] = GetParam();
  const auto actual_variant_code = legate::mapping::detail::to_variant_code(processor_kind);
  ASSERT_EQ(actual_variant_code, expected_variant_code);
}

// Test round-trip conversions to ensure consistency
TEST_F(MappingCommonTest, RoundTripTaskTargetConversions)
{
  // TaskTarget -> Processor::Kind -> TaskTarget
  for (auto task_target : {legate::mapping::TaskTarget::GPU,
                           legate::mapping::TaskTarget::OMP,
                           legate::mapping::TaskTarget::CPU}) {
    auto processor_kind         = legate::mapping::detail::to_kind(task_target);
    auto round_trip_task_target = legate::mapping::detail::to_target(processor_kind);
    ASSERT_EQ(task_target, round_trip_task_target);
  }
}

TEST_F(MappingCommonTest, RoundTripVariantCodeConversions)
{
  // VariantCode -> Processor::Kind -> VariantCode (via TaskTarget)
  for (auto variant_code :
       {legate::VariantCode::CPU, legate::VariantCode::GPU, legate::VariantCode::OMP}) {
    auto processor_kind          = legate::mapping::detail::to_kind(variant_code);
    auto task_target             = legate::mapping::detail::to_target(processor_kind);
    auto round_trip_variant_code = legate::mapping::detail::to_variant_code(task_target);
    ASSERT_EQ(variant_code, round_trip_variant_code);
  }
}

TEST_F(MappingCommonTest, RoundTripStoreTargetConversions)
{
  // StoreTarget -> Memory::Kind -> StoreTarget
  for (auto store_target : {legate::mapping::StoreTarget::SYSMEM,
                            legate::mapping::StoreTarget::FBMEM,
                            legate::mapping::StoreTarget::ZCMEM,
                            legate::mapping::StoreTarget::SOCKETMEM}) {
    auto memory_kind             = legate::mapping::detail::to_kind(store_target);
    auto round_trip_store_target = legate::mapping::detail::to_target(memory_kind);
    ASSERT_EQ(store_target, round_trip_store_target);
  }
}

// Test special Processor::Kind behavior
TEST_F(MappingCommonTest, SpecialProcessorKindHandling)
{
  // PY_PROC should map to CPU task target (as documented in implementation)
  ASSERT_EQ(legate::mapping::detail::to_target(Legion::Processor::Kind::PY_PROC),
            legate::mapping::TaskTarget::CPU);

  // LOC_PROC should also map to CPU
  ASSERT_EQ(legate::mapping::detail::to_target(Legion::Processor::Kind::LOC_PROC),
            legate::mapping::TaskTarget::CPU);
}

}  // namespace common_api_unit
