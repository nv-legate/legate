/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/dlpack/to_dlpack.h>

#include <legate.h>

#include <legate/utilities/detail/dlpack/common.h>
#include <legate/utilities/detail/dlpack/dlpack.h>
#include <legate/utilities/detail/dlpack/from_dlpack.h>

#include <fmt/format.h>

#include <numeric>
#include <unit/dlpack/common.h>
#include <utilities/utilities.h>

namespace test_to_dlpack {

class ToDLPackUnit : public DefaultFixture {};

namespace {

template <typename T>
[[nodiscard]] std::pair<legate::LogicalStore, legate::Type> make_store(const legate::Shape& shape,
                                                                       T fill_value = T{7})
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto ty             = legate::primitive_type(legate::type_code_of_v<T>);
  auto store          = runtime->create_store(shape, ty);

  runtime->issue_fill(store, legate::Scalar{fill_value});
  return {std::move(store), std::move(ty)};
}

void check_basic(
  const std::unique_ptr<DLManagedTensorVersioned, void (*)(DLManagedTensorVersioned*)>& dlpack)
{
  ASSERT_THAT(dlpack, ::testing::NotNull());
  ASSERT_EQ(dlpack->version.major, DLPACK_MAJOR_VERSION);
  ASSERT_EQ(dlpack->version.minor, DLPACK_MINOR_VERSION);
  ASSERT_THAT(dlpack->manager_ctx, ::testing::NotNull());
  ASSERT_EQ(dlpack->deleter, dlpack.get_deleter());
}

void check_device(const DLDevice& device, const DLDeviceType& expected_device_type)
{
  ASSERT_EQ(device.device_type, expected_device_type);
  ASSERT_EQ(device.device_id, 0);
}

template <typename T>
void check_dtype(const DLDataType& dtype, const legate::Type& ty)
{
  ASSERT_EQ(dtype.code, dlpack_common::to_dlpack_code<T>());
  ASSERT_EQ(dtype.bits, ty.size() * CHAR_BIT);
  ASSERT_EQ(dtype.lanes, 1);
}

void check_shape(const DLTensor& tensor, const legate::Shape& shape)
{
  const auto shape_span = std::vector<std::uint64_t>{tensor.shape, tensor.shape + tensor.ndim};
  const auto expected   = shape.extents().data();

  ASSERT_THAT(shape_span, ::testing::ContainerEq(expected));
}

void check_strides(const DLTensor& tensor)
{
  const auto strides = legate::Span<const std::int64_t>{tensor.shape, tensor.shape + tensor.ndim};

  ASSERT_THAT(strides, ::testing::ElementsAre(2, 2));
}

void check_tensor_basic(const DLTensor& tensor, const legate::LogicalStore& store)
{
  ASSERT_EQ(tensor.ndim, store.dim());
  ASSERT_EQ(tensor.byte_offset, 0);
  ASSERT_THAT(tensor.data, ::testing::NotNull());

  const auto size =
    std::reduce(tensor.shape, tensor.shape + tensor.ndim, std::int64_t{1}, std::multiplies<>{});

  ASSERT_EQ(size, store.extents().volume());
}

}  // namespace

template <typename T>
class ToDLPackUnitTyped : public ToDLPackUnit {};

TYPED_TEST_SUITE(ToDLPackUnitTyped, dlpack_common::AllTypes, dlpack_common::NameGenerator);

TYPED_TEST(ToDLPackUnitTyped, Basic)
{
  constexpr auto DIM     = 2;
  const auto FILL_VALUE  = TypeParam{7};
  const auto SET_VALUE   = TypeParam{3};
  const auto shape       = legate::Shape{2, 2};
  const auto [store, ty] = make_store<TypeParam>(shape, FILL_VALUE);
  const auto phys        = store.get_physical_store();
  const auto dlpack      = phys.to_dlpack();

  check_basic(dlpack);
  ASSERT_EQ(dlpack->flags, 0);

  const auto& tensor = dlpack->dl_tensor;

  check_tensor_basic(tensor, store);

  {
    auto* const data = static_cast<TypeParam*>(tensor.data);
    auto span        = legate::Span<TypeParam>{data, data + shape.volume()};
    const auto alloc = phys.get_inline_allocation();

    // We did not copy, so the pointer from the inline allocation should be the same as the one
    // we get from the dlpack export.
    ASSERT_EQ(alloc.ptr, span.data());
    ASSERT_THAT(span, ::testing::Each(FILL_VALUE));
    std::fill(span.begin(), span.end(), SET_VALUE);
  }

  check_device(tensor.device, /*expected_device_type*/ DLDeviceType::kDLCPU);
  check_dtype<TypeParam>(tensor.dtype, ty);
  check_shape(tensor, shape);
  check_strides(tensor);

  const auto acc = phys.template span_read_accessor<TypeParam, DIM>();

  for (legate::coord_t i = 0; i < acc.extent(0); ++i) {
    for (legate::coord_t j = 0; j < acc.extent(1); ++j) {
      // We did not copy, so this value should have changed
      ASSERT_EQ(acc(i, j), SET_VALUE);
    }
  }
}

TYPED_TEST(ToDLPackUnitTyped, MustCopy)
{
  constexpr auto DIM     = 2;
  const auto FILL_VALUE  = TypeParam{7};
  const auto shape       = legate::Shape{2, 2};
  const auto [store, ty] = make_store<TypeParam>(shape, FILL_VALUE);
  const auto phys        = store.get_physical_store();
  const auto dlpack      = phys.to_dlpack(/* copy */ true);

  check_basic(dlpack);

  ASSERT_EQ(dlpack->flags, DLPACK_FLAG_BITMASK_IS_COPIED);

  const auto& tensor = dlpack->dl_tensor;

  check_tensor_basic(tensor, store);

  {
    const auto SET_VALUE = TypeParam{3};
    auto* const data     = static_cast<TypeParam*>(tensor.data);
    auto span            = legate::Span<TypeParam>{data, data + shape.volume()};
    const auto alloc     = phys.get_inline_allocation();

    // We did copy, so the pointer from the inline allocation should NOT be the same as the one
    // we get from the dlpack export.
    ASSERT_NE(alloc.ptr, span.data());
    ASSERT_THAT(span, ::testing::Each(FILL_VALUE));
    std::fill(span.begin(), span.end(), SET_VALUE);
  }

  check_device(tensor.device, /*expected_device_type*/ DLDeviceType::kDLCPU);
  check_dtype<TypeParam>(tensor.dtype, ty);
  check_shape(tensor, shape);
  check_strides(tensor);

  const auto acc = phys.template span_read_accessor<TypeParam, DIM>();

  for (legate::coord_t i = 0; i < acc.extent(0); ++i) {
    for (legate::coord_t j = 0; j < acc.extent(1); ++j) {
      // Ensure our copy did not affect the original store
      ASSERT_EQ(acc(i, j), FILL_VALUE);
    }
  }
}

TYPED_TEST(ToDLPackUnitTyped, CopyZCMEM)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();

  if (machine.count(legate::mapping::TaskTarget::GPU) == 0) {
    GTEST_SKIP() << "Skipping test due to no GPU available";
  }

  const auto FILL_VALUE  = TypeParam{7};
  const auto shape       = legate::Shape{2, 2};
  const auto [store, ty] = make_store<TypeParam>(shape, FILL_VALUE);
  const auto phys        = store.get_physical_store(legate::mapping::StoreTarget::ZCMEM);
  const auto dlpack      = phys.to_dlpack(/* copy */ true);

  check_basic(dlpack);

  ASSERT_EQ(dlpack->flags, DLPACK_FLAG_BITMASK_IS_COPIED);

  const auto& tensor = dlpack->dl_tensor;

  check_tensor_basic(tensor, store);

  {
    auto* const data = static_cast<TypeParam*>(tensor.data);
    const auto alloc = phys.get_inline_allocation();

    // We did copy, so the pointer from the inline allocation should NOT be the same as the one
    // we get from the dlpack export.
    ASSERT_NE(alloc.ptr, data);
  }

  check_device(tensor.device, /*expected_device_type*/ DLDeviceType::kDLCUDAHost);
  check_dtype<TypeParam>(tensor.dtype, ty);
  check_shape(tensor, shape);
  check_strides(tensor);
}

TYPED_TEST(ToDLPackUnitTyped, CopyFBMEM)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  if (machine.count(legate::mapping::TaskTarget::GPU) == 0) {
    GTEST_SKIP() << "Skipping test due to no GPU available";
  }

  const auto FILL_VALUE  = TypeParam{7};
  const auto shape       = legate::Shape{2, 2};
  const auto [store, ty] = make_store<TypeParam>(shape, FILL_VALUE);
  const auto phys        = store.get_physical_store(legate::mapping::StoreTarget::FBMEM);
  const auto dlpack      = phys.to_dlpack(/* copy */ true);

  check_basic(dlpack);

  ASSERT_EQ(dlpack->flags, DLPACK_FLAG_BITMASK_IS_COPIED);

  const auto& tensor = dlpack->dl_tensor;

  check_tensor_basic(tensor, store);

  {
    auto* const data = static_cast<TypeParam*>(tensor.data);
    const auto alloc = phys.get_inline_allocation();

    // We did copy, so the pointer from the inline allocation should NOT be the same as the one
    // we get from the dlpack export.
    ASSERT_NE(alloc.ptr, data);
  }

  check_device(tensor.device, /*expected_device_type*/ DLDeviceType::kDLCUDA);
  check_dtype<TypeParam>(tensor.dtype, ty);
  check_shape(tensor, shape);
  check_strides(tensor);
}

TYPED_TEST(ToDLPackUnitTyped, NeverCopy)
{
  constexpr auto DIM     = 2;
  const auto FILL_VALUE  = TypeParam{7};
  const auto SET_VALUE   = TypeParam{3};
  const auto shape       = legate::Shape{2, 2};
  const auto [store, ty] = make_store<TypeParam>(shape, FILL_VALUE);
  const auto phys        = store.get_physical_store();
  const auto dlpack      = phys.to_dlpack(/* copy */ false);

  check_basic(dlpack);

  ASSERT_EQ(dlpack->flags, 0);

  const auto& tensor = dlpack->dl_tensor;

  check_tensor_basic(tensor, store);

  {
    auto* const data = static_cast<TypeParam*>(tensor.data);
    auto span        = legate::Span<TypeParam>{data, data + shape.volume()};
    const auto alloc = phys.get_inline_allocation();

    // We did NOT copy, so the pointer from the inline allocation should be the same as the one
    // we get from the dlpack export.
    ASSERT_EQ(alloc.ptr, span.data());
    ASSERT_THAT(span, ::testing::Each(FILL_VALUE));
    std::fill(span.begin(), span.end(), SET_VALUE);
  }

  check_device(tensor.device, /*expected_device_type*/ DLDeviceType::kDLCPU);
  check_dtype<TypeParam>(tensor.dtype, ty);
  check_shape(tensor, shape);
  check_strides(tensor);

  const auto acc = phys.template span_read_accessor<TypeParam, DIM>();

  for (legate::coord_t i = 0; i < acc.extent(0); ++i) {
    for (legate::coord_t j = 0; j < acc.extent(1); ++j) {
      // We did NOT copy, so this should be our new value
      ASSERT_EQ(acc(i, j), SET_VALUE);
    }
  }
}

class ToDLPackUnitVersion
  : public ToDLPackUnit,
    public ::testing::WithParamInterface<std::pair<DLPackVersion, DLPackVersion>> {};

INSTANTIATE_TEST_SUITE_P(,
                         ToDLPackUnitVersion,
                         ::testing::Values(std::make_pair(DLPackVersion{1, 0}, DLPackVersion{1, 0}),
                                           std::make_pair(DLPackVersion{1, 1}, DLPackVersion{1, 1}),
                                           std::make_pair(DLPackVersion{99, 99},
                                                          DLPackVersion{DLPACK_MAJOR_VERSION,
                                                                        DLPACK_MINOR_VERSION})));

static_assert(DLPACK_MAJOR_VERSION == 1 &&  //  NOLINT(misc-redundant-expression)
                DLPACK_MINOR_VERSION == 1,  //  NOLINT(misc-redundant-expression)
              "Update test values above to include new version");

TEST_P(ToDLPackUnitVersion, Supported)
{
  const auto [ver, expected] = GetParam();
  const auto [store, _]      = make_store<std::int32_t>(legate::Shape{1});
  const auto phys            = store.get_physical_store();
  const auto dlpack =
    legate::detail::to_dlpack(phys, /* copy */ std::nullopt, /* stream */ std::nullopt, ver);

  ASSERT_EQ(dlpack->version.major, expected.major);
  ASSERT_EQ(dlpack->version.minor, expected.minor);
}

class ToDLPackUnitUnsupportedLegateTypeInput
  : public ToDLPackUnit,
    public ::testing::WithParamInterface<legate::Type::Code> {};

INSTANTIATE_TEST_SUITE_P(ToDLPackUnitLegateType,
                         ToDLPackUnitUnsupportedLegateTypeInput,
                         ::testing::ValuesIn(dlpack_common::get_unsupported_legate_types()));

TEST_P(ToDLPackUnitUnsupportedLegateTypeInput, Unsupported)
{
  auto create_store = [&](legate::Type::Code code) {
    auto* const runtime = legate::Runtime::get_runtime();

    const auto ty = [&]() -> legate::Type {
      if (code == legate::Type::Code::FIXED_ARRAY) {
        return legate::fixed_array_type(legate::int64(), /*N=*/2);
      }
      if (code == legate::Type::Code::STRUCT) {
        return legate::struct_type({legate::int64(), legate::int64()});
      }
      return legate::null_type();
    }();

    return runtime->create_store(legate::Shape{0}, ty);
  };

  auto store      = create_store(GetParam());
  const auto phys = store.get_physical_store();

  ASSERT_THAT(
    [&] {
      std::ignore = legate::detail::to_dlpack(
        phys, /* copy */ std::nullopt, /* stream */ std::nullopt, /* max_version */ std::nullopt);
    },
    ::testing::ThrowsMessage<std::invalid_argument>(::testing::ContainsRegex(
      fmt::format("Conversion of .* type to DLPack is not yet supported"))));
}

TEST_F(ToDLPackUnit, BadVersion)
{
  const auto [store, _] = make_store<std::int32_t>(legate::Shape{1});
  const auto phys       = store.get_physical_store();

  constexpr auto BAD_VERSION = DLPackVersion{0, 4};

  ASSERT_THAT(
    [&] {
      std::ignore = legate::detail::to_dlpack(
        phys, /* copy */ std::nullopt, /* stream */ std::nullopt, BAD_VERSION);
    },
    ::testing::ThrowsMessage<std::runtime_error>(
      ::testing::HasSubstr(fmt::format("Cannot satisfy request for DLPack tensor of version {}.{}",
                                       BAD_VERSION.major,
                                       BAD_VERSION.minor))));
}

TEST_F(ToDLPackUnit, BadDevice)
{
  const auto [store, _] = make_store<std::int32_t>(legate::Shape{1});
  const auto phys       = store.get_physical_store();

  constexpr auto BAD_DEVICE = DLDevice{DLDeviceType::kDLROCM, 7};

  ASSERT_THAT(
    [&] {
      std::ignore = legate::detail::to_dlpack(phys,
                                              /* copy */ std::nullopt,
                                              /* stream */ std::nullopt,
                                              /* max_version */ std::nullopt,
                                              BAD_DEVICE);
    },
    ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
      fmt::format("Cannot satisfy request to provide DLPack tensor on device (device_type {}, "
                  "device_id {}). This task would provide a tensor on device (device_type {}, "
                  "device_id {}) instead.",
                  BAD_DEVICE.device_type,
                  BAD_DEVICE.device_id,
                  DLDeviceType::kDLCPU,
                  0))));
}

TEST_F(ToDLPackUnit, BadTypeSize)
{
  // Test the case where the type size exceeds the maximum value of std::uint8_t
  constexpr auto BINARY_SIZE =
    static_cast<std::size_t>(std::numeric_limits<std::uint8_t>::max() + 1);
  auto binary_ty = legate::binary_type(BINARY_SIZE);

  auto runtime    = legate::Runtime::get_runtime();
  auto store      = runtime->create_store(legate::Shape{1}, binary_ty);
  const auto phys = store.get_physical_store();

  ASSERT_THAT(
    [&] {
      std::ignore = legate::detail::to_dlpack(phys,
                                              /* copy */ std::nullopt,
                                              /* stream */ std::nullopt,
                                              /* max_version */ std::nullopt,
                                              /* device */ DLDevice{DLDeviceType::kDLCPU, 0});
    },
    ::testing::ThrowsMessage<std::domain_error>(
      ::testing::ContainsRegex("Cannot convert Legate type .* to DLPack type. "
                               "The number of bits required to represent an element of the type .* "
                               "> max size of DLPack bits datatype .*")));
}

TEST_F(ToDLPackUnit, BinaryTypeNeverCopy)
{
  auto runtime     = legate::Runtime::get_runtime();
  const auto shape = legate::Shape{2, 2};
  auto store       = runtime->create_store(shape, legate::binary_type(8));
  const auto phys  = store.get_physical_store();

  const auto dlpack = phys.to_dlpack();

  check_basic(dlpack);
  ASSERT_EQ(dlpack->flags, 0);

  const auto& tensor = dlpack->dl_tensor;

  check_tensor_basic(tensor, store);
  check_device(tensor.device, /*expected_device_type*/ DLDeviceType::kDLCPU);
  ASSERT_EQ(tensor.dtype.code, DLDataTypeCode::kDLOpaqueHandle);
  check_shape(tensor, shape);
  check_strides(tensor);
}

TEST_F(ToDLPackUnit, BinaryTypeMustCopy)
{
  auto runtime     = legate::Runtime::get_runtime();
  const auto shape = legate::Shape{2, 2};
  auto store       = runtime->create_store(shape, legate::binary_type(8));
  const auto phys  = store.get_physical_store();

  const auto dlpack = phys.to_dlpack(/* copy */ true);

  check_basic(dlpack);
  ASSERT_EQ(dlpack->flags, DLPACK_FLAG_BITMASK_IS_COPIED);

  const auto& tensor = dlpack->dl_tensor;

  check_tensor_basic(tensor, store);
  check_device(tensor.device, /*expected_device_type*/ DLDeviceType::kDLCPU);
  ASSERT_EQ(tensor.dtype.code, DLDataTypeCode::kDLOpaqueHandle);
  check_shape(tensor, shape);
  check_strides(tensor);
}

TEST_F(ToDLPackUnit, Release)
{
  const auto [store, _] = make_store<std::int32_t>(legate::Shape{1});
  const auto phys       = store.get_physical_store();
  auto dlpack           = phys.to_dlpack();
  auto* const ptr       = dlpack.release();

  ASSERT_NO_THROW(ptr->deleter(ptr));
}

}  // namespace test_to_dlpack
