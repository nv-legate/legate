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
#include <legate/utilities/detail/dlpack/common.h>
#include <legate/utilities/detail/dlpack/dlpack.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <fmt/format.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace formatter_test {

constexpr std::int32_t SCALAR_VALUE = 42;

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

class FormatOperationKind : public DefaultFixture,
                            public ::testing::WithParamInterface<
                              std::tuple<legate::detail::Operation::Kind, std::string_view>> {};

class FormatImgComputationHint : public DefaultFixture,
                                 public ::testing::WithParamInterface<
                                   std::tuple<legate::ImageComputationHint, std::string_view>> {};

class FormatTaskTarget : public DefaultFixture,
                         public ::testing::WithParamInterface<
                           std::tuple<legate::mapping::TaskTarget, std::string_view>> {};

class FormatStoreTarget : public DefaultFixture,
                          public ::testing::WithParamInterface<
                            std::tuple<legate::mapping::StoreTarget, std::string_view>> {};

class FormatDLPackTypeCode
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<DLDataTypeCode, std::string_view>> {};

class FormatDLPackDeviceType
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<DLDeviceType, std::string_view>> {};

class FormatLegionPrivilegeMode
  : public DefaultFixture,
    public ::testing::WithParamInterface<std::tuple<legion_privilege_mode_t, std::string_view>> {};

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
  FormatOperationKind,
  ::testing::Values(
    std::make_tuple(legate::detail::Operation::Kind::ATTACH, "Attach"),
    std::make_tuple(legate::detail::Operation::Kind::AUTO_TASK, "AutoTask"),
    std::make_tuple(legate::detail::Operation::Kind::COPY, "Copy"),
    std::make_tuple(legate::detail::Operation::Kind::DISCARD, "Discard"),
    std::make_tuple(legate::detail::Operation::Kind::EXECUTION_FENCE, "ExecutionFence"),
    std::make_tuple(legate::detail::Operation::Kind::FILL, "Fill"),
    std::make_tuple(legate::detail::Operation::Kind::GATHER, "Gather"),
    std::make_tuple(legate::detail::Operation::Kind::INDEX_ATTACH, "IndexAttach"),
    std::make_tuple(legate::detail::Operation::Kind::MANUAL_TASK, "ManualTask"),
    std::make_tuple(legate::detail::Operation::Kind::MAPPING_FENCE, "MappingFence"),
    std::make_tuple(legate::detail::Operation::Kind::REDUCE, "Reduce"),
    std::make_tuple(legate::detail::Operation::Kind::RELEASE_REGION_FIELD, "ReleaseRegionField"),
    std::make_tuple(legate::detail::Operation::Kind::SCATTER, "Scatter"),
    std::make_tuple(legate::detail::Operation::Kind::SCATTER_GATHER, "ScatterGather"),
    std::make_tuple(legate::detail::Operation::Kind::TIMING, "Timing")));

INSTANTIATE_TEST_SUITE_P(
  FormatterUnit,
  FormatImgComputationHint,
  ::testing::Values(std::make_tuple(legate::ImageComputationHint::NO_HINT, "NO_HINT"),
                    std::make_tuple(legate::ImageComputationHint::MIN_MAX, "MIN_MAX"),
                    std::make_tuple(legate::ImageComputationHint::FIRST_LAST, "FIRST_LAST")));

INSTANTIATE_TEST_SUITE_P(FormatterUnit,
                         FormatTaskTarget,
                         ::testing::Values(std::make_tuple(legate::mapping::TaskTarget::CPU, "CPU"),
                                           std::make_tuple(legate::mapping::TaskTarget::GPU, "GPU"),
                                           std::make_tuple(legate::mapping::TaskTarget::OMP,
                                                           "OMP")));

INSTANTIATE_TEST_SUITE_P(
  FormatterUnit,
  FormatStoreTarget,
  ::testing::Values(std::make_tuple(legate::mapping::StoreTarget::SYSMEM, "SYSMEM"),
                    std::make_tuple(legate::mapping::StoreTarget::FBMEM, "FBMEM"),
                    std::make_tuple(legate::mapping::StoreTarget::ZCMEM, "ZCMEM"),
                    std::make_tuple(legate::mapping::StoreTarget::SOCKETMEM, "SOCKETMEM")));

INSTANTIATE_TEST_SUITE_P(
  FormatterUnit,
  FormatDLPackTypeCode,
  ::testing::Values(std::make_tuple(DLDataTypeCode::kDLInt, "Int"),
                    std::make_tuple(DLDataTypeCode::kDLUInt, "UInt"),
                    std::make_tuple(DLDataTypeCode::kDLFloat, "Float"),
                    std::make_tuple(DLDataTypeCode::kDLOpaqueHandle, "OpaqueHandle"),
                    std::make_tuple(DLDataTypeCode::kDLBfloat, "Bfloat"),
                    std::make_tuple(DLDataTypeCode::kDLComplex, "Complex"),
                    std::make_tuple(DLDataTypeCode::kDLBool, "Bool"),
                    std::make_tuple(DLDataTypeCode::kDLFloat8_e3m4, "Float8_e3m4"),
                    std::make_tuple(DLDataTypeCode::kDLFloat8_e4m3, "Float8_e4m3"),
                    std::make_tuple(DLDataTypeCode::kDLFloat8_e4m3b11fnuz, "Float8_e4m3b11fnuz"),
                    std::make_tuple(DLDataTypeCode::kDLFloat8_e4m3fn, "Float8_e4m3fn"),
                    std::make_tuple(DLDataTypeCode::kDLFloat8_e4m3fnuz, "Float8_e4m3fnuz"),
                    std::make_tuple(DLDataTypeCode::kDLFloat8_e5m2, "Float8_e5m2"),
                    std::make_tuple(DLDataTypeCode::kDLFloat8_e5m2fnuz, "Float8_e5m2fnuz"),
                    std::make_tuple(DLDataTypeCode::kDLFloat8_e8m0fnu, "Float8_e8m0fnu"),
                    std::make_tuple(DLDataTypeCode::kDLFloat6_e2m3fn, "Float6_e2m3fn"),
                    std::make_tuple(DLDataTypeCode::kDLFloat6_e3m2fn, "Float6_e3m2fn"),
                    std::make_tuple(DLDataTypeCode::kDLFloat4_e2m1fn, "Float4_e2m1fn")));

INSTANTIATE_TEST_SUITE_P(FormatterUnit,
                         FormatDLPackDeviceType,
                         ::testing::Values(std::make_tuple(DLDeviceType::kDLCPU, "CPU"),
                                           std::make_tuple(DLDeviceType::kDLCUDA, "CUDA"),
                                           std::make_tuple(DLDeviceType::kDLCUDAHost, "CUDAHost"),
                                           std::make_tuple(DLDeviceType::kDLOpenCL, "OpenCL"),
                                           std::make_tuple(DLDeviceType::kDLVulkan, "Vulkan"),
                                           std::make_tuple(DLDeviceType::kDLMetal, "Metal"),
                                           std::make_tuple(DLDeviceType::kDLVPI, "VPI"),
                                           std::make_tuple(DLDeviceType::kDLROCM, "ROCM"),
                                           std::make_tuple(DLDeviceType::kDLROCMHost, "ROCMHost"),
                                           std::make_tuple(DLDeviceType::kDLExtDev, "ExtDev"),
                                           std::make_tuple(DLDeviceType::kDLCUDAManaged,
                                                           "CUDAManaged"),
                                           std::make_tuple(DLDeviceType::kDLOneAPI, "OneAPI"),
                                           std::make_tuple(DLDeviceType::kDLWebGPU, "WebGPU"),
                                           std::make_tuple(DLDeviceType::kDLHexagon, "Hexagon"),
                                           std::make_tuple(DLDeviceType::kDLMAIA, "MAIA")));

INSTANTIATE_TEST_SUITE_P(
  FormatterUnit,
  FormatLegionPrivilegeMode,
  ::testing::Values(std::make_tuple(LEGION_READ_ONLY, "LEGION_READ_ONLY"),
                    std::make_tuple(LEGION_READ_DISCARD, "LEGION_READ_DISCARD"),
                    std::make_tuple(LEGION_REDUCE, "LEGION_REDUCE"),
                    std::make_tuple(LEGION_WRITE_ONLY, "LEGION_WRITE_ONLY"),
                    std::make_tuple(LEGION_READ_WRITE, "LEGION_READ_WRITE"),
                    std::make_tuple(LEGION_WRITE_DISCARD, "LEGION_WRITE_DISCARD"),
                    std::make_tuple(LEGION_WRITE_PRIV, "LEGION_WRITE_PRIV"),
                    std::make_tuple(LEGION_NO_ACCESS, "LEGION_NO_ACCESS"),
                    std::make_tuple(LEGION_DISCARD_MASK, "LEGION_DISCARD_MASK"),
                    std::make_tuple(LEGION_DISCARD_OUTPUT_MASK, "LEGION_DISCARD_OUTPUT_MASK")));

template <typename>
using FormatID = ::testing::Test;

using IDTypeList = ::testing::
  Types<legate::LocalTaskID, legate::GlobalTaskID, legate::LocalRedopID, legate::GlobalRedopID>;

TYPED_TEST_SUITE(FormatID, IDTypeList, );

TEST_P(FormatType, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatShape, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatVariantCode, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatOperationKind, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatImgComputationHint, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatTaskTarget, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatStoreTarget, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatDLPackTypeCode, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatDLPackDeviceType, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_P(FormatLegionPrivilegeMode, Basic)
{
  const auto& [format_obj, expect_result] = GetParam();

  ASSERT_EQ(fmt::format("{}", format_obj), expect_result);
}

TEST_F(FormatterUnit, InvalidDLPackTypeCode)
{
  if (LEGATE_DEFINED(LEGATE_HAS_ASAN)) {
    GTEST_SKIP() << "Skipping test due to exceed enum range of DLDataTypeCode";
  }

  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)
  auto type_code = static_cast<DLDataTypeCode>(99);  // Invalid data type code
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)

  ASSERT_EQ(fmt::format("{}", type_code), "Unknown DLPack data type");
}

TEST_F(FormatterUnit, InvalidDLPackDeviceType)
{
  if (LEGATE_DEFINED(LEGATE_HAS_ASAN)) {
    GTEST_SKIP() << "Skipping test due to exceed enum range of DLDeviceType";
  }

  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)
  auto device_type = static_cast<DLDeviceType>(99);  // Invalid device type
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)

  ASSERT_EQ(fmt::format("{}", device_type), "Unknown DLPack device type");
}

TEST_F(FormatterUnit, ExecutionFence)
{
  // To hit format operation
  const legate::InternalSharedPtr<legate::detail::Operation> smart_ptr{
    new legate::detail::ExecutionFence{/*unique_id=*/1, /*block=*/false}};
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

TEST_F(FormatterUnit, TaskInfo)
{
  const auto task_info = legate::TaskInfo{"test_task"};

  ASSERT_EQ(fmt::format("{}", task_info), "test_task {}");
}

TEST_F(FormatterUnit, BoundLogicalStore)
{
  const auto runtime           = legate::Runtime::get_runtime();
  const auto bound_store       = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  const auto& bound_store_impl = bound_store.impl();

  ASSERT_THAT(
    fmt::format("{}", *bound_store_impl),
    ::testing::MatchesRegex(
      R"(Store\([0-9]+\) \{shape: \[1\], type: int32, storage: Storage\([0-9]+\) \{kind: Future, level: [0-9]+\}\})"));
}

TEST_F(FormatterUnit, UnboundLogicalStore)
{
  const auto runtime             = legate::Runtime::get_runtime();
  const auto unbound_store       = runtime->create_store(legate::int64());
  const auto& unbound_store_impl = unbound_store.impl();

  ASSERT_THAT(
    fmt::format("{}", *unbound_store_impl),
    ::testing::MatchesRegex(
      R"(Store\([0-9]+\) \{shape: \(unbound\), type: int64, storage: Storage\([0-9]+\) \{kind: Region, level: [0-9]+, region: unbound\}\})"));
}

TEST_F(FormatterUnit, TransformedBoundLogicalStore)
{
  const auto runtime        = legate::Runtime::get_runtime();
  const auto bound_store    = runtime->create_store(legate::Scalar{SCALAR_VALUE});
  const auto promoted       = bound_store.promote(/*extra_dim=*/0, /*dim_size=*/5);
  const auto& promoted_impl = promoted.impl();

  ASSERT_THAT(
    fmt::format("{}", *promoted_impl),
    ::testing::MatchesRegex(
      R"(Store\([0-9]+\) \{shape: \[5, 1\], transform: Promote\(extra_dim: 0, dim_size: 5\), type: int32, storage: Storage\([0-9]+\) \{kind: Future, level: [0-9]+\}\})"));
}

TYPED_TEST(FormatID, Basic)
{
  constexpr auto id = TypeParam{0};

  ASSERT_EQ(fmt::format("{}", id), "0");
}

}  // namespace formatter_test
