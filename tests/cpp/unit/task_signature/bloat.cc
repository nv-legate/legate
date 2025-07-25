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

namespace test_task_signature_bloat {

namespace {

enum TaskIDs : std::uint8_t {
  BLOAT_SINGLE_INPUT_SINGLE_OUTPUT,
  BLOAT_SINGLE_INPUT_ALL_OUTPUTS,
  BLOAT_ALL_INPUTS_SINGLE_OUTPUT,
  BLOAT_ALL_INPUTS_ALL_OUTPUTS,
  BLOAT_SINGLE_INPUT_SINGLE_INPUT,
};

[[nodiscard]] legate::LogicalArray make_array()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto ret      = runtime->create_array(legate::Shape{3}, legate::int32());

  runtime->issue_fill(ret, legate::Scalar{0});
  return ret;
}

class BloatSingleInputSingleOutput : public legate::LegateTask<BloatSingleInputSingleOutput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{BLOAT_SINGLE_INPUT_SINGLE_OUTPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::bloat(legate::proxy::inputs[0], legate::proxy::outputs[0], {0}, {0})}}));

  static void cpu_variant(legate::TaskContext) {}
};

class BloatSingleInputAllOutputs : public legate::LegateTask<BloatSingleInputAllOutputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{BLOAT_SINGLE_INPUT_ALL_OUTPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::bloat(legate::proxy::inputs[0], legate::proxy::outputs, {0}, {0})}}));

  static void cpu_variant(legate::TaskContext) {}
};

class BloatAllInputsSingleOutput : public legate::LegateTask<BloatAllInputsSingleOutput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{BLOAT_ALL_INPUTS_SINGLE_OUTPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::bloat(legate::proxy::inputs, legate::proxy::outputs[0], {0}, {0})}}));

  static void cpu_variant(legate::TaskContext) {}
};

class BloatAllInputsAllOutputs : public legate::LegateTask<BloatAllInputsAllOutputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{BLOAT_ALL_INPUTS_ALL_OUTPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::bloat(legate::proxy::inputs, legate::proxy::outputs, {0}, {0})}}));

  static void cpu_variant(legate::TaskContext) {}
};

class BloatSingleInputSingleInput : public legate::LegateTask<BloatSingleInputSingleInput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{BLOAT_SINGLE_INPUT_SINGLE_INPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::bloat(legate::proxy::inputs[0], legate::proxy::inputs[0], {0}, {0})}}));

  static void cpu_variant(legate::TaskContext) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_signature_bloat";

  static void registration_callback(legate::Library library)
  {
    BloatSingleInputSingleOutput::register_variants(library);
    BloatSingleInputAllOutputs::register_variants(library);
    BloatAllInputsSingleOutput::register_variants(library);
    BloatAllInputsAllOutputs::register_variants(library);
    BloatSingleInputSingleInput::register_variants(library);
  }
};

}  // namespace

using BloatTesterTypeList = ::testing::Types<BloatSingleInputSingleOutput,
                                             BloatSingleInputAllOutputs,
                                             BloatAllInputsSingleOutput,
                                             BloatAllInputsAllOutputs,
                                             BloatSingleInputSingleInput>;

template <typename T>
class TaskSignatureBloatUnit : public RegisterOnceFixture<Config> {};

TYPED_TEST_SUITE(TaskSignatureBloatUnit, BloatTesterTypeList, );

TYPED_TEST(TaskSignatureBloatUnit, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, TypeParam::TASK_CONFIG.task_id());

  task.add_input(make_array());
  task.add_output(make_array());
  runtime->submit(std::move(task));
}

}  // namespace test_task_signature_bloat
