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

namespace test_task_signature_image {

namespace {

enum TaskIDs : std::uint8_t {
  IMAGE_SINGLE_INPUT_SINGLE_OUTPUT,
  IMAGE_SINGLE_INPUT_ALL_OUTPUTS,
  IMAGE_ALL_INPUTS_SINGLE_OUTPUT,
  IMAGE_ALL_INPUTS_ALL_OUTPUTS,
  IMAGE_SINGLE_INPUT_SINGLE_INPUT,
};

[[nodiscard]] legate::LogicalArray make_array()
{
  auto* runtime = legate::Runtime::get_runtime();
  auto ret      = runtime->create_array(legate::Shape{3}, legate::point_type(1));

  auto data_store = ret.data();
  auto accessor   = data_store.get_physical_store().write_accessor<legate::Point<1>, 1>();
  accessor[0]     = legate::Point<1>{0};
  accessor[1]     = legate::Point<1>{1};
  accessor[2]     = legate::Point<1>{2};

  return ret;
}

class ImageSingleInputSingleOutput : public legate::LegateTask<ImageSingleInputSingleOutput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{IMAGE_SINGLE_INPUT_SINGLE_OUTPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::image(legate::proxy::inputs[0], legate::proxy::outputs[0])}}));

  static void cpu_variant(legate::TaskContext) {}
};

class ImageSingleInputAllOutputs : public legate::LegateTask<ImageSingleInputAllOutputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{IMAGE_SINGLE_INPUT_ALL_OUTPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::image(legate::proxy::inputs[0], legate::proxy::outputs)}}));

  static void cpu_variant(legate::TaskContext) {}
};

class ImageAllInputsSingleOutput : public legate::LegateTask<ImageAllInputsSingleOutput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{IMAGE_ALL_INPUTS_SINGLE_OUTPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::image(legate::proxy::inputs, legate::proxy::outputs[0])}}));

  static void cpu_variant(legate::TaskContext) {}
};

class ImageAllInputsAllOutputs : public legate::LegateTask<ImageAllInputsAllOutputs> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{IMAGE_ALL_INPUTS_ALL_OUTPUTS}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints({{legate::image(
        legate::proxy::inputs, legate::proxy::outputs, legate::ImageComputationHint::NO_HINT)}}));

  static void cpu_variant(legate::TaskContext) {}
};

class ImageSingleInputSingleInput : public legate::LegateTask<ImageSingleInputSingleInput> {
 public:
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{IMAGE_SINGLE_INPUT_SINGLE_INPUT}}.with_signature(
      legate::TaskSignature{}.inputs(1).outputs(1).constraints(
        {{legate::image(legate::proxy::inputs[0], legate::proxy::inputs[0])}}));

  static void cpu_variant(legate::TaskContext) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_signature_image";

  static void registration_callback(legate::Library library)
  {
    ImageSingleInputSingleOutput::register_variants(library);
    ImageSingleInputAllOutputs::register_variants(library);
    ImageAllInputsSingleOutput::register_variants(library);
    ImageAllInputsAllOutputs::register_variants(library);
    ImageSingleInputSingleInput::register_variants(library);
  }
};

}  // namespace

using ImageTesterTypeList = ::testing::Types<ImageSingleInputSingleOutput,
                                             ImageSingleInputAllOutputs,
                                             ImageAllInputsSingleOutput,
                                             ImageAllInputsAllOutputs,
                                             ImageSingleInputSingleInput>;

template <typename T>
class TaskSignatureImageUnit : public RegisterOnceFixture<Config> {};

TYPED_TEST_SUITE(TaskSignatureImageUnit, ImageTesterTypeList, );

TYPED_TEST(TaskSignatureImageUnit, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(library, TypeParam::TASK_CONFIG.task_id());

  task.add_input(make_array());
  task.add_output(make_array());
  runtime->submit(std::move(task));
}

}  // namespace test_task_signature_image
