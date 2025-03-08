/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/shape.h>
#include <legate/operation/detail/execution_fence.h>
#include <legate/operation/detail/operation.h>
#include <legate/partitioning/detail/constraint.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace formatter_test {

// Dummy task to make the runtime think the store is initialized
struct FormatterBaseTask : public legate::LegateTask<FormatterBaseTask> {
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext /*context*/) {}
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_formatter";
  static void registration_callback(legate::Library library)
  {
    FormatterBaseTask::register_variants(library);
  }
};

class FormatterUnit : public RegisterOnceFixture<Config> {};

class FormatType
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::string_view>> {};

class FormatShape
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::detail::Shape, std::string_view>> {};

class FormatVariantCode
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legate::VariantCode, std::string_view>> {};

class FormatImgComputationHint : public DefaultFixture,
                                 public ::testing::WithParamInterface<
                                   std::tuple<legate::ImageComputationHint, std::string_view>> {};

INSTANTIATE_TEST_SUITE_P(
  FormatterUnit,
  FormatType,
  ::testing::Values(std::make_tuple(legate::bool_(), "bool"),
                    std::make_tuple(legate::int8(), "int8"),
                    std::make_tuple(legate::int16(), "int16"),
                    std::make_tuple(legate::int32(), "int32"),
                    std::make_tuple(legate::int64(), "int64"),
                    std::make_tuple(legate::uint8(), "uint8"),
                    std::make_tuple(legate::uint16(), "uint16"),
                    std::make_tuple(legate::uint32(), "uint32"),
                    std::make_tuple(legate::uint64(), "uint64"),
                    std::make_tuple(legate::float16(), "float16"),
                    std::make_tuple(legate::float32(), "float32"),
                    std::make_tuple(legate::float64(), "float64"),
                    std::make_tuple(legate::complex64(), "complex64"),
                    std::make_tuple(legate::complex128(), "complex128"),
                    std::make_tuple(legate::point_type(2), "int64[2]"),
                    std::make_tuple(legate::rect_type(2), "{int64[2]:0,int64[2]:16}"),
                    std::make_tuple(legate::null_type(), "null_type"),
                    std::make_tuple(legate::string_type(), "string"),
                    std::make_tuple(legate::binary_type(1), "binary(1)"),
                    std::make_tuple(legate::list_type(legate::int8()), "list(int8)")));

INSTANTIATE_TEST_SUITE_P(
  FormatterUnit,
  FormatShape,
  ::testing::Values(std::make_tuple(legate::detail::Shape{2}, "Shape(unbound 2D)"),
                    std::make_tuple(legate::detail::Shape{{1, 2}}, "Shape [1, 2]")));

INSTANTIATE_TEST_SUITE_P(FormatterUnit,
                         FormatVariantCode,
                         ::testing::Values(std::make_tuple(legate::VariantCode::CPU, "CPU_VARIANT"),
                                           std::make_tuple(legate::VariantCode::GPU, "GPU_VARIANT"),
                                           std::make_tuple(legate::VariantCode::OMP,
                                                           "OMP_VARIANT")));

INSTANTIATE_TEST_SUITE_P(
  FormatterUnit,
  FormatImgComputationHint,
  ::testing::Values(std::make_tuple(legate::ImageComputationHint::NO_HINT, "NO_HINT"),
                    std::make_tuple(legate::ImageComputationHint::MIN_MAX, "MIN_MAX"),
                    std::make_tuple(legate::ImageComputationHint::FIRST_LAST, "FIRST_LAST")));

template <typename>
using FormatID = ::testing::Test;

using IDTypeList = ::testing::
  Types<legate::LocalTaskID, legate::GlobalTaskID, legate::LocalRedopID, legate::GlobalRedopID>;

TYPED_TEST_SUITE(FormatID, IDTypeList, );

TEST_P(FormatType, Basic)
{
  auto& [format_obj, expect_result] = GetParam();
  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatShape, Basic)
{
  auto& [format_obj, expect_result] = GetParam();
  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatVariantCode, Basic)
{
  auto& [format_obj, expect_result] = GetParam();
  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatImgComputationHint, Basic)
{
  auto& [format_obj, expect_result] = GetParam();
  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_F(FormatterUnit, ExecutionFence)
{
  // To hit format operation
  const legate::InternalSharedPtr<legate::detail::Operation> smart_ptr{
    new legate::detail::ExecutionFence{1, false}};
  constexpr std::string_view expect_str = "ExecutionFence:1";
  ASSERT_EQ(fmt::format("{}", *smart_ptr), expect_str);
}

TEST_F(FormatterUnit, Alignment)
{
  // To hit format constraint and variable
  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(Config::LIBRARY_NAME);
  auto task    = runtime->create_task(context, FormatterBaseTask::TASK_CONFIG.task_id());

  auto part1 = task.declare_partition();
  auto part2 = task.declare_partition();
  ASSERT_THAT(fmt::format("{}", *(part1.impl())),
              ::testing::MatchesRegex(R"(X0\{formatter_test::FormatterBaseTask:[0-9]+\})"));
  ASSERT_THAT(fmt::format("{}", *(part2.impl())),
              ::testing::MatchesRegex(R"(X1\{formatter_test::FormatterBaseTask:[0-9]+\})"));

  auto alignment = legate::detail::align(part1.impl(), part2.impl());
  ASSERT_THAT(
    fmt::format("{}", *alignment),
    ::testing::MatchesRegex(
      R"(Align\(X0\{formatter_test::FormatterBaseTask:[0-9]+\}, X1\{formatter_test::FormatterBaseTask:[0-9]+\}\))"));
}

TYPED_TEST(FormatID, Basic)
{
  constexpr auto id = TypeParam{0};
  ASSERT_EQ(fmt::format("{}", id), "0");
}

}  // namespace formatter_test
