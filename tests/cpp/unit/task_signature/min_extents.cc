/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/partitioning/constraint.h>
#include <legate/partitioning/proxy.h>
#include <legate/task/task_signature.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <utilities/utilities.h>

namespace test_task_signature_min_extents {

namespace {

enum TaskIDs : std::uint8_t {
  MIN_EXTENTS_ALL_INPUTS,
};

constexpr std::int32_t SENTINEL_VALUE = 17;
constexpr auto MIN_EXTENT             = std::uint64_t{3};
constexpr auto NUM_INPUTS             = std::uint32_t{2};
constexpr auto NUM_OUTPUTS            = std::uint32_t{1};

[[nodiscard]] legate::LogicalStore make_store()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto ret      = runtime->create_store(legate::Shape{3}, legate::int32());

  runtime->issue_fill(ret, legate::Scalar{0});
  return ret;
}

class MinExtentsAllInputs : public legate::LegateTask<MinExtentsAllInputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp, bugprone-throwing-static-initialization)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{MIN_EXTENTS_ALL_INPUTS}}.with_signature(
      legate::TaskSignature{}
        .inputs(NUM_INPUTS)
        .outputs(NUM_OUTPUTS)
        .constraints({{legate::min_extents(legate::proxy::inputs,
                                           legate::tuple<std::uint64_t>{MIN_EXTENT})}}));

  static void cpu_variant(legate::TaskContext context)
  {
    const auto input0_shape  = context.input(0).shape<1>();
    const auto input1_shape  = context.input(1).shape<1>();
    const auto input0_extent = input0_shape.hi[0] - input0_shape.lo[0] + 1;
    const auto input1_extent = input1_shape.hi[0] - input1_shape.lo[0] + 1;

    ASSERT_GE(input0_extent, MIN_EXTENT);
    ASSERT_GE(input1_extent, MIN_EXTENT);

    auto output = context.output(0);
    auto shape  = output.shape<1>();
    auto acc    = output.write_accessor<std::int32_t, 1>(shape);

    for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
      acc[*it] = SENTINEL_VALUE;
    }
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_signature_min_extents";

  static void registration_callback(legate::Library library)
  {
    MinExtentsAllInputs::register_variants(library);
  }
};

class TaskSignatureMinExtentsUnit : public RegisterOnceFixture<Config> {};

}  // namespace

TEST_F(TaskSignatureMinExtentsUnit, AllInputs)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, MinExtentsAllInputs::TASK_CONFIG.task_id());
  auto output  = make_store();

  task.add_input(make_store());
  task.add_input(make_store());
  task.add_output(output);
  runtime->submit(std::move(task));

  const auto store = output.get_physical_store();
  const auto acc   = store.read_accessor<std::int32_t, 1>();

  for (legate::PointInRectIterator<1> it{store.shape<1>()}; it.valid(); ++it) {
    ASSERT_EQ(acc[*it], SENTINEL_VALUE);
  }
}

}  // namespace test_task_signature_min_extents
