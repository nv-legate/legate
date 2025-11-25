/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/dlpack/dlpack.h>
#include <legate/utilities/detail/dlpack/from_dlpack.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <numeric>
#include <unit/dlpack/common.h>
#include <utilities/utilities.h>

namespace test_from_dlpack_versioned {

class FromDLPackVersionedUnit : public DefaultFixture {};

TEST_F(FromDLPackVersionedUnit, BadVersion)
{
  constexpr auto BAD_VERSION = DLPACK_MAJOR_VERSION + 1;
  DLManagedTensorVersioned dlm_tensor{};

  dlm_tensor.version.major = BAD_VERSION;

  ASSERT_THAT(
    [&] {
      auto* ptr   = &dlm_tensor;
      std::ignore = legate::detail::from_dlpack(&ptr);
    },
    ::testing::ThrowsMessage<std::runtime_error>(::testing::HasSubstr(
      fmt::format("Unsupported major version for DLPack tensor {} (Legate only supports {}.x)",
                  BAD_VERSION,
                  DLPACK_MAJOR_VERSION))));
}

TEST_F(FromDLPackVersionedUnit, BadDim)
{
  constexpr auto BAD_DIM = LEGATE_MAX_DIM + 1;
  DLManagedTensorVersioned dlm_tensor{};

  dlm_tensor.version.major  = DLPACK_MAJOR_VERSION;
  dlm_tensor.dl_tensor.ndim = BAD_DIM;

  ASSERT_THAT(
    [&] {
      auto* ptr   = &dlm_tensor;
      std::ignore = legate::detail::from_dlpack(&ptr);
    },
    ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr(fmt::format(
      "DLPack tensor dimension {} exceeds LEGATE_MAX_DIM (0, {}), recompile Legate with a "
      "larger maximum dimension to support this conversion",
      BAD_DIM,
      LEGATE_MAX_DIM))));
}

TEST_F(FromDLPackVersionedUnit, NULLPointer)
{
  DLManagedTensorVersioned** dlm_tensor_ptr_ptr = nullptr;
  DLManagedTensorVersioned* dlm_tensor_ptr      = nullptr;

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(dlm_tensor_ptr_ptr); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("DLManagedTensorVersioned** argument must not be NULL")));

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(&dlm_tensor_ptr); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("DLManagedTensorVersioned* must not be NULL")));
}

namespace {

template <typename T>
void dltensor_deleter(DLManagedTensorVersioned* self)
{
  ASSERT_EQ(self->manager_ctx, nullptr);
  delete[] static_cast<T*>(std::exchange(self->dl_tensor.data, nullptr));
  delete[] std::exchange(self->dl_tensor.shape, nullptr);
  delete[] std::exchange(self->dl_tensor.strides, nullptr);
  delete self;
}

// This returns a raw pointer because from_dlpack() takes ownership of the tensor (whose
// deleter will delete this instance.)
template <typename T, std::size_t DIM = 3>
[[nodiscard]] DLManagedTensorVersioned* make_tensor()
{
  constexpr auto PER_DIM_SHAPE = 2;
  constexpr auto SIZE          = 1 << DIM;  // 2^DIM

  auto uniq        = std::make_unique<DLManagedTensorVersioned>(DLManagedTensorVersioned{});
  auto& dlm_tensor = *uniq;

  dlm_tensor.version.major = DLPACK_MAJOR_VERSION;
  dlm_tensor.version.major = DLPACK_MINOR_VERSION;
  dlm_tensor.manager_ctx   = nullptr;
  dlm_tensor.deleter       = dltensor_deleter<T>;
  dlm_tensor.flags         = 0;

  auto& tensor = dlm_tensor.dl_tensor;

  tensor.data = [] {
    auto tmp = std::make_unique<T[]>(SIZE);

    std::iota(tmp.get(), tmp.get() + SIZE, 1);
    return tmp.release();
  }();
  tensor.device = DLDevice{DLDeviceType::kDLCPU, 0};
  tensor.ndim   = DIM;
  tensor.dtype  = DLDataType{static_cast<std::uint8_t>(dlpack_common::to_dlpack_code<T>()),
                            /* bits */ sizeof(T) * CHAR_BIT,
                            /* lanes */ 1};

  tensor.shape = [&] {
    auto shape = std::make_unique<std::int64_t[]>(DIM);

    std::fill(shape.get(), shape.get() + DIM, PER_DIM_SHAPE);
    return shape.release();
  }();

  tensor.strides = [] {
    auto tmp = std::make_unique<std::int64_t[]>(DIM);

    static_assert(DIM == 3);
    tmp[0] = 1;
    tmp[1] = 2;
    tmp[2] = 4;
    return tmp.release();
  }();

  tensor.byte_offset = 0;

  return uniq.release();
}

template <typename T, std::size_t DIM = 3>
void check_store(const legate::LogicalStore& store, const DLTensor& tensor)
{
  ASSERT_EQ(store.dim(), tensor.ndim);
  ASSERT_EQ(store.type(), legate::primitive_type(legate::type_code_of_v<T>));
  ASSERT_EQ(store.type().size(), tensor.dtype.bits / CHAR_BIT);

  const auto shape = store.shape();

  const auto expected_volume = legate::detail::array_volume(
    legate::Span<const std::int64_t>{tensor.shape, tensor.shape + tensor.ndim});

  ASSERT_EQ(shape.volume(), expected_volume);

  auto&& extents = shape.extents();

  ASSERT_THAT(extents.data(), ::testing::ElementsAreArray(tensor.shape, tensor.ndim));

  {
    const auto phys = store.get_physical_store();
    const auto acc  = phys.template span_read_accessor<T, DIM>();
    T value         = T{1};

    static_assert(DIM == 3, "Need to fix up below check for DIM != 3");
    for (legate::coord_t k = 0; k < acc.extent(2); ++k) {
      for (legate::coord_t j = 0; j < acc.extent(1); ++j) {
        for (legate::coord_t i = 0; i < acc.extent(0); ++i) {
          ASSERT_EQ(acc(i, j, k), value) << "i " << i << ", j " << j << ", k " << k;
          value += T{1};
        }
      }
    }
  }
}

}  // namespace

TEST_F(FromDLPackVersionedUnit, BadLanes)
{
  constexpr auto BAD_LANES = 10;
  using data_type          = std::int64_t;
  auto* dlm_tensor         = make_tensor<data_type>();

  dlm_tensor->dl_tensor.dtype.lanes = BAD_LANES;

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(&dlm_tensor); },
              ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(fmt::format(
                "Conversion from multi-lane packed vector types (code Int, bits {}, lanes {}) to "
                "Legate not yet supported",
                sizeof(data_type) * CHAR_BIT,
                BAD_LANES))));
}

template <typename T>
class FromDLPackVersionedUnitBadTypeSize : public FromDLPackVersionedUnit {};

TYPED_TEST_SUITE(FromDLPackVersionedUnitBadTypeSize,
                 dlpack_common::AllTypes,
                 dlpack_common::NameGenerator);

TYPED_TEST(FromDLPackVersionedUnitBadTypeSize, Basic)
{
  constexpr auto BAD_BITS = 123;
  auto* dlm_tensor        = make_tensor<TypeParam>();

  dlm_tensor->dl_tensor.dtype.bits = BAD_BITS;

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(&dlm_tensor); },
              ::testing::ThrowsMessage<std::out_of_range>(::testing::ContainsRegex(
                fmt::format("Number of bits {} for .*type is not supported", BAD_BITS))));
}

class FromDLPackVersionedUnitUnsupportedTyped : public FromDLPackVersionedUnit {};

class VersionedUnsupportedInput : public FromDLPackVersionedUnit,
                                  public ::testing::WithParamInterface<DLDataTypeCode> {};

INSTANTIATE_TEST_SUITE_P(FromDLPackVersionedUnitUnsupportedTyped,
                         VersionedUnsupportedInput,
                         ::testing::ValuesIn(dlpack_common::get_unsupported_data_type_codes()));

TEST_P(VersionedUnsupportedInput, Basic)
{
  auto* dlm_tensor = make_tensor<float>();

  dlm_tensor->dl_tensor.dtype.code = GetParam();

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(&dlm_tensor); },
              ::testing::ThrowsMessage<std::invalid_argument>(::testing::ContainsRegex(
                "Conversion of DLPack type code .* to Legate not yet supported")));
}

TEST_F(FromDLPackVersionedUnit, BadTypeSizeBool)
{
  constexpr auto BAD_BITS = 123;
  auto* dlm_tensor        = make_tensor<bool>();

  dlm_tensor->dl_tensor.dtype.bits = BAD_BITS;

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(&dlm_tensor); },
              ::testing::ThrowsMessage<std::out_of_range>(::testing::HasSubstr(fmt::format(
                "Cannot represent boolean data type whose size in bytes != {} (have {})",
                sizeof(bool),
                BAD_BITS / CHAR_BIT))));
}

TEST_F(FromDLPackVersionedUnit, NonContiguousStrides)
{
  constexpr auto BIG_STRIDE = 300;
  auto* dlm_tensor          = make_tensor<bool>();

  dlm_tensor->dl_tensor.strides[0] = BIG_STRIDE;

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(&dlm_tensor); },
              ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(
                "Conversion of non-contiguous strided tensors is not yet supported")));
}

TEST_F(FromDLPackVersionedUnit, NonMonotonousStrides)
{
  auto* dlm_tensor = make_tensor<bool>();
  auto* strides    = dlm_tensor->dl_tensor.strides;
  const auto strides_span =
    legate::Span<const std::int64_t>{strides, strides + dlm_tensor->dl_tensor.ndim};

  std::swap(strides[0], strides[1]);

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(&dlm_tensor); },
              ::testing::ThrowsMessage<std::invalid_argument>(::testing::HasSubstr(fmt::format(
                "Cannot represent non monotonous {} strides as Legate dimension ordering",
                fmt::join(strides_span, ", ")))));
}

template <typename T>
class FromDLPackVersionedUnitTyped : public FromDLPackVersionedUnit {};

TYPED_TEST_SUITE(FromDLPackVersionedUnitTyped,
                 dlpack_common::AllTypes,
                 dlpack_common::NameGenerator);

TYPED_TEST(FromDLPackVersionedUnitTyped, Basic)
{
  constexpr auto DIM = 3;

  auto* dlm_tensor   = make_tensor<TypeParam, DIM>();
  const auto& tensor = dlm_tensor->dl_tensor;
  auto store         = legate::detail::from_dlpack(&dlm_tensor);

  check_store<TypeParam, DIM>(store, tensor);
}

TEST_F(FromDLPackVersionedUnit, OpaqueDataType)
{
  constexpr auto DIM = 3;
  auto* dlm_tensor   = make_tensor<std::int8_t, DIM>();

  dlm_tensor->dl_tensor.dtype.code = DLDataTypeCode::kDLOpaqueHandle;

  const auto& tensor = dlm_tensor->dl_tensor;
  auto store         = legate::detail::from_dlpack(&dlm_tensor);

  ASSERT_EQ(store.dim(), tensor.ndim);
  ASSERT_EQ(store.type().code(), legate::Type::Code::BINARY);
  ASSERT_EQ(store.type().size(), tensor.dtype.bits / CHAR_BIT);

  const auto shape = store.shape();
  auto&& extents   = shape.extents();

  ASSERT_EQ(shape.volume(), 1 << DIM);
  ASSERT_THAT(extents.data(), ::testing::ElementsAreArray(tensor.shape, tensor.ndim));
}

TEST_F(FromDLPackVersionedUnit, EmptyDim)
{
  constexpr auto DIM = 3;
  using data_type    = std::int8_t;

  auto* dlm_tensor = make_tensor<data_type, DIM>();

  dlm_tensor->dl_tensor.shape[1] = 0;

  const auto& tensor = dlm_tensor->dl_tensor;
  auto store         = legate::detail::from_dlpack(&dlm_tensor);

  check_store<data_type, DIM>(store, tensor);
}

class FromDLPackVersionedUnitDeviceType : public FromDLPackVersionedUnit {};

class FromDLPackVersionedCPUInput : public FromDLPackVersionedUnit,
                                    public ::testing::WithParamInterface<DLDeviceType> {};

class FromDLPackVersionedGPUInput : public FromDLPackVersionedUnit,
                                    public ::testing::WithParamInterface<DLDeviceType> {};

class FromDLPackVersionedUnsupportedDeviceInput
  : public FromDLPackVersionedUnit,
    public ::testing::WithParamInterface<DLDeviceType> {};

INSTANTIATE_TEST_SUITE_P(FromDLPackVersionedUnitDeviceType,
                         FromDLPackVersionedCPUInput,
                         ::testing::ValuesIn(dlpack_common::get_cpu_device_types()));

INSTANTIATE_TEST_SUITE_P(FromDLPackVersionedUnitDeviceType,
                         FromDLPackVersionedGPUInput,
                         ::testing::ValuesIn(dlpack_common::get_gpu_device_types()));

INSTANTIATE_TEST_SUITE_P(FromDLPackVersionedUnitDeviceType,
                         FromDLPackVersionedUnsupportedDeviceInput,
                         ::testing::ValuesIn(dlpack_common::get_unsupported_device_types()));

TEST_P(FromDLPackVersionedCPUInput, Basic)
{
  constexpr auto DIM = 3;
  using data_type    = std::int8_t;

  auto* dlm_tensor = make_tensor<data_type, DIM>();

  dlm_tensor->dl_tensor.device.device_type = GetParam();

  const auto& tensor = dlm_tensor->dl_tensor;
  auto store         = legate::detail::from_dlpack(&dlm_tensor);

  check_store<data_type, DIM>(store, tensor);
}

TEST_P(FromDLPackVersionedGPUInput, Basic)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();

  if (machine.count(legate::mapping::TaskTarget::GPU) == 0) {
    GTEST_SKIP() << "Skipping test due to no GPU available";
  }

  constexpr auto DIM = 3;
  using data_type    = std::int8_t;

  auto* dlm_tensor = make_tensor<data_type, DIM>();

  dlm_tensor->dl_tensor.device.device_type = GetParam();

  const auto& tensor = dlm_tensor->dl_tensor;
  auto store         = legate::detail::from_dlpack(&dlm_tensor);

  check_store<data_type, DIM>(store, tensor);
}

TEST_P(FromDLPackVersionedUnsupportedDeviceInput, Basic)
{
  auto* dlm_tensor = make_tensor<std::int8_t>();

  dlm_tensor->dl_tensor.device.device_type = GetParam();

  ASSERT_THAT([&] { std::ignore = legate::detail::from_dlpack(&dlm_tensor); },
              ::testing::ThrowsMessage<std::invalid_argument>(::testing::ContainsRegex(
                "Conversion from DLPack device type .* to Legate store is not yet supported.")));
}

using FromDLPackVersionedDeathTest = FromDLPackVersionedUnit;

TEST_F(FromDLPackVersionedDeathTest, InvalidDataType)
{
  // Skip this test if LEGATE_USE_DEBUG is not defined
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skipping test due to verifying logic in LEGATE_ASSERT()";
  }

  auto* dlm_tensor     = make_tensor<std::int8_t>();
  auto uniq_dlm_tensor = std::unique_ptr<DLManagedTensorVersioned, std::function<void(void*)>>{
    dlm_tensor, [tptr = dlm_tensor](void*) {
      if (tptr->deleter) {
        tptr->deleter(tptr);
      }
    }};

  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)
  uniq_dlm_tensor->dl_tensor.dtype.code =
    static_cast<DLDataTypeCode>(99);  // Invalid data type code
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)

  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
  auto* raw_ptr = uniq_dlm_tensor.get();
  ASSERT_DEATH(static_cast<void>(legate::detail::from_dlpack(&raw_ptr)),
               ::testing::AnyOf(
                 // Normal exception raised
                 ::testing::HasSubstr("Unhandled DLPack type code"),
                 // Error from UBSAN, if we are running with it
                 ::testing::HasSubstr("runtime error: load of value 99")));
}

TEST_F(FromDLPackVersionedDeathTest, InvalidDeviceType)
{
  // Skip this test if LEGATE_USE_DEBUG is not defined
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Skipping test due to verifying logic in LEGATE_ASSERT()";
  }

  auto* dlm_tensor     = make_tensor<std::int8_t>();
  auto uniq_dlm_tensor = std::unique_ptr<DLManagedTensorVersioned, std::function<void(void*)>>{
    dlm_tensor, [tptr = dlm_tensor](void*) {
      if (tptr->deleter) {
        tptr->deleter(tptr);
      }
    }};

  // NOLINTBEGIN(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)
  uniq_dlm_tensor->dl_tensor.device.device_type =
    static_cast<DLDeviceType>(99);  // Invalid device type
  // NOLINTEND(clang-analyzer-optin.core.EnumCastOutOfRange,readability-magic-numbers)

  auto* raw_ptr = uniq_dlm_tensor.get();
  ASSERT_DEATH(static_cast<void>(legate::detail::from_dlpack(&raw_ptr)),
               ::testing::AnyOf(
                 // Normal exception raised
                 ::testing::HasSubstr("Unhandled device type"),
                 // Error from UBSAN, if we are running with it
                 ::testing::HasSubstr("runtime error: load of value 99")));
}

TEST_F(FromDLPackVersionedDeathTest, ZeroDim)
{
  constexpr auto DIM  = 3;
  auto* const runtime = legate::Runtime::get_runtime();
  auto* dlm_tensor    = make_tensor<std::int8_t, DIM>();
  const auto uniq_dlm_tensor =
    std::unique_ptr<DLManagedTensorVersioned, std::function<void(void*)>>{
      dlm_tensor, [tptr = dlm_tensor](void*) {
        if (tptr->deleter) {
          tptr->deleter(tptr);
        }
      }};
  auto* raw_ptr = uniq_dlm_tensor.get();

  dlm_tensor->dl_tensor.ndim = 0;

  ASSERT_DEATH(
    [&] {
      std::ignore = legate::detail::from_dlpack(&raw_ptr);
      // The assertion in question is triggered by launching the Attach operation (to attach to
      // the external allocation). If the execution window is large enough (really, >1) the
      // attachment doesn't trigger until a manual flush.
      runtime->issue_execution_fence(/* block */ true);
    }(),
    // The LEGATE_CHECK() in generate_legion_dims() failed
    ::testing::HasSubstr("assertion failed: ndim > 0"));
}

}  // namespace test_from_dlpack_versioned
