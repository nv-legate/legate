/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

namespace test_task_signature_alignment {

namespace {

enum TaskIDs : std::uint8_t {
  ALIGNMENT_SINGLE_INPUT_SINGLE_OUTPUT,
  ALIGNMENT_SINGLE_INPUT_ALL_OUTPUTS,
  ALIGNMENT_ALL_INPUTS_SINGLE_OUTPUT,
  ALIGNMENT_ALL_INPUTS_ALL_OUTPUTS,
  ALIGNMENT_ALL_INPUTS_ALL_REDUCTIONS,
};

[[nodiscard]] legate::LogicalArray make_array()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto ret      = runtime->create_array(legate::Shape{3}, legate::int32());

  runtime->issue_fill(ret, legate::Scalar{0});
  return ret;
}

class AlignmentSingleInputSingleOutput
  : public legate::LegateTask<AlignmentSingleInputSingleOutput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{ALIGNMENT_SINGLE_INPUT_SINGLE_OUTPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::align(legate::proxy::inputs[0], legate::proxy::outputs[0])}}));

  static void cpu_variant(legate::TaskContext) {}
};

class AlignmentSingleInputAllOutputs : public legate::LegateTask<AlignmentSingleInputAllOutputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{ALIGNMENT_SINGLE_INPUT_ALL_OUTPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::align(legate::proxy::inputs[0], legate::proxy::outputs)}}));

  static void cpu_variant(legate::TaskContext) {}
};

class AlignmentAllInputsSingleOutput : public legate::LegateTask<AlignmentAllInputsSingleOutput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{ALIGNMENT_ALL_INPUTS_SINGLE_OUTPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::align(legate::proxy::inputs, legate::proxy::outputs[0])}}));

  static void cpu_variant(legate::TaskContext) {}
};

class AlignmentAllInputsAllOutputs : public legate::LegateTask<AlignmentAllInputsAllOutputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{ALIGNMENT_ALL_INPUTS_ALL_OUTPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::align(legate::proxy::inputs, legate::proxy::outputs)}}));

  static void cpu_variant(legate::TaskContext) {}
};

class AlignmentAllInputsAllReductions : public legate::LegateTask<AlignmentAllInputsAllReductions> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{ALIGNMENT_ALL_INPUTS_ALL_REDUCTIONS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).redops(1).constraints(
        {{legate::align(legate::proxy::inputs, legate::proxy::reductions)}}));

  static void cpu_variant(legate::TaskContext) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_signature_alignment";

  static void registration_callback(legate::Library library)
  {
    AlignmentSingleInputSingleOutput::register_variants(library);
    AlignmentSingleInputAllOutputs::register_variants(library);
    AlignmentAllInputsSingleOutput::register_variants(library);
    AlignmentAllInputsAllOutputs::register_variants(library);
    AlignmentAllInputsAllReductions::register_variants(library);
  }
};

}  // namespace

using AlignmentTesterTypeList = ::testing::Types<AlignmentSingleInputSingleOutput,
                                                 AlignmentSingleInputAllOutputs,
                                                 AlignmentAllInputsSingleOutput,
                                                 AlignmentAllInputsAllOutputs,
                                                 AlignmentAllInputsAllReductions>;

template <typename T>
class TaskSignatureAlignmentUnit : public RegisterOnceFixture<Config> {};

TYPED_TEST_SUITE(TaskSignatureAlignmentUnit, AlignmentTesterTypeList, );

TYPED_TEST(TaskSignatureAlignmentUnit, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, TypeParam::TASK_CONFIG.task_id());

  task.add_input(make_array());
  task.add_output(make_array());
  task.add_reduction(make_array(), legate::ReductionOpKind::ADD);
  runtime->submit(std::move(task));
}

}  // namespace test_task_signature_alignment
