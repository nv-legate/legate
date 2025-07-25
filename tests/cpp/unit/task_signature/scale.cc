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

namespace test_task_signature_scale {

namespace {

enum TaskIDs : std::uint8_t {
  SCALE_SINGLE_INPUT_SINGLE_OUTPUT,
  SCALE_SINGLE_INPUT_ALL_OUTPUTS,
  SCALE_ALL_INPUTS_SINGLE_OUTPUT,
  SCALE_ALL_INPUTS_ALL_OUTPUTS,
  SCALE_SINGLE_INPUT_SINGLE_INPUT,
};

[[nodiscard]] legate::LogicalArray make_array()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto ret      = runtime->create_array(legate::Shape{3}, legate::int32());

  runtime->issue_fill(ret, legate::Scalar{0});
  return ret;
}

class ScaleSingleInputSingleOutput : public legate::LegateTask<ScaleSingleInputSingleOutput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{SCALE_SINGLE_INPUT_SINGLE_OUTPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::scale({2}, legate::proxy::inputs[0], legate::proxy::outputs[0])}}));

  static void cpu_variant(legate::TaskContext) {}
};

class ScaleSingleInputAllOutputs : public legate::LegateTask<ScaleSingleInputAllOutputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{SCALE_SINGLE_INPUT_ALL_OUTPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::scale({2}, legate::proxy::inputs[0], legate::proxy::outputs)}}));

  static void cpu_variant(legate::TaskContext) {}
};

class ScaleAllInputsSingleOutput : public legate::LegateTask<ScaleAllInputsSingleOutput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{SCALE_ALL_INPUTS_SINGLE_OUTPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::scale({2}, legate::proxy::inputs, legate::proxy::outputs[0])}}));

  static void cpu_variant(legate::TaskContext) {}
};

class ScaleAllInputsAllOutputs : public legate::LegateTask<ScaleAllInputsAllOutputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{SCALE_ALL_INPUTS_ALL_OUTPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::scale({2}, legate::proxy::inputs, legate::proxy::outputs)}}));

  static void cpu_variant(legate::TaskContext) {}
};

class ScaleSingleInputSingleInput : public legate::LegateTask<ScaleSingleInputSingleInput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{SCALE_SINGLE_INPUT_SINGLE_INPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::scale({2}, legate::proxy::inputs, legate::proxy::inputs)}}));

  static void cpu_variant(legate::TaskContext) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_signature_scale";

  static void registration_callback(legate::Library library)
  {
    ScaleSingleInputSingleOutput::register_variants(library);
    ScaleSingleInputAllOutputs::register_variants(library);
    ScaleAllInputsSingleOutput::register_variants(library);
    ScaleAllInputsAllOutputs::register_variants(library);
    ScaleSingleInputSingleInput::register_variants(library);
  }
};

}  // namespace

using ScaleTesterTypeList = ::testing::Types<ScaleSingleInputSingleOutput,
                                             ScaleSingleInputAllOutputs,
                                             ScaleAllInputsSingleOutput,
                                             ScaleAllInputsAllOutputs,
                                             ScaleSingleInputSingleInput>;

template <typename T>
class TaskSignatureScaleUnit : public RegisterOnceFixture<Config> {};

TYPED_TEST_SUITE(TaskSignatureScaleUnit, ScaleTesterTypeList, );

TYPED_TEST(TaskSignatureScaleUnit, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, TypeParam::TASK_CONFIG.task_id());

  task.add_input(make_array());
  task.add_output(make_array());
  runtime->submit(std::move(task));
}

}  // namespace test_task_signature_scale
