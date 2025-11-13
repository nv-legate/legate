/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/core_mapper.h>

#include <legate.h>

#include <gtest/gtest.h>

#include <memory>
#include <utilities/utilities.h>

namespace core_mapper_allocation_test {

namespace {

class ImageTestConfig {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_find_bounding_box";
  static void registration_callback(legate::Library library);
};

constexpr std::uint8_t DUMMY_IMAGE_TASK = 0;

// A simple task that takes Point input and produces output
// This will be used with image constraint to trigger FIND_BOUNDING_BOX
template <std::int32_t DIM>
class DummyImageTask : public legate::LegateTask<DummyImageTask<DIM>> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{DUMMY_IMAGE_TASK + DIM}};

  static void cpu_variant(legate::TaskContext context)
  {
    // Simple no-op task - we just want to trigger the image constraint processing
    // Note: both are inputs to avoid non-disjoint partition write issues
    auto domain_input = context.input(0).data();
    auto range_input  = context.input(1).data();

    // Just read the data (no-op task for triggering image constraint)
    auto domain_shape = domain_input.shape<1>();
    auto range_shape  = range_input.shape<DIM>();

    if (domain_shape.empty() || range_shape.empty()) {
      return;
    }

    // Read accessors to satisfy Legion requirements
    static_cast<void>(domain_input.read_accessor<legate::Point<DIM>, 1>());
    static_cast<void>(range_input.read_accessor<std::int32_t, DIM>());
  }
};

void ImageTestConfig::registration_callback(legate::Library library)
{
  DummyImageTask<2>::register_variants(library);
}

using CoreMapperTest      = DefaultFixture;
using CoreMapperDeathTest = CoreMapperTest;

// Fixture that also registers the image test library
class CoreMapperImageConstraintTest : public RegisterOnceFixture<ImageTestConfig> {};

}  // namespace

TEST_F(CoreMapperTest, CreateCoreMapper)
{
  auto mapper = legate::mapping::detail::create_core_mapper();

  ASSERT_NE(mapper, nullptr);
}

TEST_F(CoreMapperDeathTest, RecordingMapperDelegation)
{
  auto mapper = legate::mapping::detail::create_core_mapper();

  ASSERT_DEATH(
    { static_cast<void>(mapper->tunable_value(legate::TunableID{0})); },
    "Tunable values are no longer supported");
}

// Test that actually triggers FIND_BOUNDING_BOX by using image constraint
TEST_F(CoreMapperImageConstraintTest, FindBoundingBoxAllocationPoolSize)
{
  auto* runtime              = legate::Runtime::get_runtime();
  auto context               = runtime->find_library(ImageTestConfig::LIBRARY_NAME);
  constexpr std::int32_t DIM = 2;

  // Create a larger Point store to force actual partitioning
  // Use enough data to trigger partition creation
  const std::uint64_t num_points = 1000;
  const std::uint64_t range_size = 100;
  auto domain_store              = runtime->create_store({num_points}, legate::point_type(DIM));

  // Initialize with varied point data to create non-trivial bounding boxes
  constexpr std::int32_t SHPAE_SIZE = 50;
  runtime->issue_fill(domain_store, legate::Scalar{legate::Point<DIM>{SHPAE_SIZE, SHPAE_SIZE}});

  // Create range store - larger to make partitioning meaningful
  auto range_store = runtime->create_store({range_size, range_size}, legate::int32());
  runtime->issue_fill(range_store, legate::Scalar{std::int32_t{0}});

  // Create task with image constraint
  // Note: Both stores are added as inputs because image partitions may be non-disjoint,
  // and Legion doesn't allow writing to non-disjoint partitions
  auto task = runtime->create_task(context, legate::LocalTaskID{DUMMY_IMAGE_TASK + DIM});

  auto part_domain = task.declare_partition();
  auto part_range  = task.declare_partition();

  task.add_input(domain_store, part_domain);
  task.add_input(range_store, part_range);

  // Add image constraint with MIN_MAX hint to trigger FIND_BOUNDING_BOX
  // Note: NO_HINT uses precise image partition and won't create FIND_BOUNDING_BOX tasks
  task.add_constraint(
    legate::image(part_domain, part_range, legate::ImageComputationHint::MIN_MAX));
  runtime->submit(std::move(task));
  runtime->issue_execution_fence(/*block=*/true);
}

}  // namespace core_mapper_allocation_test
