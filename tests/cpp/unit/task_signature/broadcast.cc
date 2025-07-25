/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/partitioning/constraint.h>
#include <legate/partitioning/proxy.h>
#include <legate/task/task_signature.h>

#include <gtest/gtest.h>

#include <stdexcept>
#include <utilities/utilities.h>

namespace test_task_signature_broadcast {

namespace {

enum TaskIDs : std::uint8_t {
  BROADCAST_SINGLE_INPUT,
  BROADCAST_ALL_INPUTS,
};

[[nodiscard]] legate::LogicalArray make_array()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto ret      = runtime->create_array(legate::Shape{3}, legate::int32());

  runtime->issue_fill(ret, legate::Scalar{0});
  return ret;
}

class BroadcastSingleInput : public legate::LegateTask<BroadcastSingleInput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{BROADCAST_SINGLE_INPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::broadcast(legate::proxy::inputs[0])}}));

  static void cpu_variant(legate::TaskContext) {}
};

class BroadcastAllInputs : public legate::LegateTask<BroadcastAllInputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{BROADCAST_ALL_INPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::broadcast(legate::proxy::inputs, legate::tuple<std::uint32_t>{0})}}));

  static void cpu_variant(legate::TaskContext) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_signature_broadcast";

  static void registration_callback(legate::Library library)
  {
    BroadcastSingleInput::register_variants(library);
    BroadcastAllInputs::register_variants(library);
  }
};

}  // namespace

using BroadcastTesterTypeList = ::testing::Types<BroadcastSingleInput, BroadcastAllInputs>;

template <typename T>
class TaskSignatureBroadcastUnit : public RegisterOnceFixture<Config> {};

TYPED_TEST_SUITE(TaskSignatureBroadcastUnit, BroadcastTesterTypeList, );

TYPED_TEST(TaskSignatureBroadcastUnit, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, TypeParam::TASK_CONFIG.task_id());

  task.add_input(make_array());
  task.add_output(make_array());
  runtime->submit(std::move(task));
}

}  // namespace test_task_signature_broadcast
