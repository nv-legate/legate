/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

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
                 dlpack_common::SignedIntTypes,
                 dlpack_common::NameGenerator);

TYPED_TEST(FromDLPackVersionedUnitTyped, Basic)
{
  constexpr auto DIM       = 3;
  constexpr auto TYPE_CODE = legate::type_code_of_v<TypeParam>;

  auto* dlm_tensor   = make_tensor<TypeParam, DIM>();
  const auto& tensor = dlm_tensor->dl_tensor;
  auto store         = legate::detail::from_dlpack(&dlm_tensor);

  ASSERT_EQ(store.dim(), tensor.ndim);
  ASSERT_EQ(store.type(), legate::primitive_type(TYPE_CODE));
  ASSERT_EQ(store.type().size(), tensor.dtype.bits / CHAR_BIT);

  const auto shape = store.shape();
  auto&& extents   = shape.extents();

  ASSERT_EQ(shape.volume(), 1 << DIM);
  ASSERT_THAT(extents.data(), ::testing::ElementsAreArray(tensor.shape, tensor.ndim));

  {
    const auto phys = store.get_physical_store();
    const auto acc  = phys.template span_read_accessor<TypeParam, DIM>();
    TypeParam value = 1;

    for (legate::coord_t k = 0; k < acc.extent(2); ++k) {
      for (legate::coord_t j = 0; j < acc.extent(1); ++j) {
        for (legate::coord_t i = 0; i < acc.extent(0); ++i) {
          ASSERT_EQ(acc(i, j, k), value) << "i " << i << ", j " << j << ", k " << k;
          ++value;
        }
      }
    }
  }
}

}  // namespace test_from_dlpack_versioned
