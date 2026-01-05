/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/dlpack/from_dlpack.h>

#include <legate/data/external_allocation.h>
#include <legate/data/shape.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/dlpack/common.h>
#include <legate/utilities/detail/dlpack/dlpack.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/zip.h>
#include <legate/utilities/span.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

namespace legate::detail {

namespace {

[[nodiscard]] legate::Shape dl_tensor_shape(const DLTensor& tensor)
{
  return legate::Shape{std::vector<std::uint64_t>{tensor.shape, tensor.shape + tensor.ndim}};
}

[[nodiscard]] legate::Type dl_tensor_type(const DLDataType& dtype)
{
  // Per DLPacks own documentation for the .code member:
  //
  // > We keep it std::uint8_t instead of DLDataTypeCode for minimal memory footprint, but the
  // > value should be one of DLDataTypeCode enum values.
  //
  // Hence we cast to silence compilers complaining about "switch over non-enum type without
  // default may not cover all cases".
  const auto code = static_cast<DLDataTypeCode>(dtype.code);

  if (dtype.lanes != 1) {
    throw TracedException<std::invalid_argument>{
      fmt::format("Conversion from multi-lane packed vector types (code {}, bits {}, lanes {}) to "
                  "Legate not yet supported",
                  code,
                  dtype.bits,
                  dtype.lanes)};
  }

  // NOLINTBEGIN(readability-magic-numbers)
  switch (code) {
    case DLDataTypeCode::kDLInt: {
      switch (const auto bits = dtype.bits) {
        case 8: return legate::int8();
        case 16: return legate::int16();
        case 32: return legate::int32();
        case 64: return legate::int64();
        default:  // legate-lint: no-switch-default
          throw TracedException<std::out_of_range>{
            fmt::format("Number of bits {} for signed integer type is not supported", bits)};
      }
    }
    case DLDataTypeCode::kDLUInt: {
      switch (const auto bits = dtype.bits) {
        case 8: return legate::uint8();
        case 16: return legate::uint16();
        case 32: return legate::uint32();
        case 64: return legate::uint64();
        default:  // legate-lint: no-switch-default
          throw TracedException<std::out_of_range>{
            fmt::format("Number of bits {} for unsigned integer type is not supported", bits)};
      }
    }
    case DLDataTypeCode::kDLFloat: [[fallthrough]];
    case DLDataTypeCode::kDLBfloat: {
      switch (const auto bits = dtype.bits) {
        case 16: return legate::float16();
        case 32: return legate::float32();
        case 64: return legate::float64();
        default:  // legate-lint: no-switch-default
          throw TracedException<std::out_of_range>{
            fmt::format("Number of bits {} for floating-point type is not supported", bits)};
      }
    }
    case DLDataTypeCode::kDLOpaqueHandle: return legate::binary_type(dtype.bits / CHAR_BIT);
    case DLDataTypeCode::kDLComplex: {
      switch (const auto bits = dtype.bits) {
        case 64: return legate::complex64();
        case 128: return legate::complex128();
        default:  // legate-lint: no-switch-default
          throw TracedException<std::out_of_range>{
            fmt::format("Number of bits {} for complex type is not supported", bits)};
      }
    }
    case DLDataTypeCode::kDLBool: {
      auto ret = legate::bool_();

      if (dtype.bits != (ret.size() * CHAR_BIT)) {
        throw TracedException<std::out_of_range>{
          fmt::format("Cannot represent boolean data type whose size in bytes != {} (have {})",
                      ret.size(),
                      dtype.bits / CHAR_BIT)};
      }
      return ret;
    }
    case DLDataTypeCode::kDLFloat8_e3m4: [[fallthrough]];
    case DLDataTypeCode::kDLFloat8_e4m3: [[fallthrough]];
    case DLDataTypeCode::kDLFloat8_e4m3b11fnuz: [[fallthrough]];
    case DLDataTypeCode::kDLFloat8_e4m3fn: [[fallthrough]];
    case DLDataTypeCode::kDLFloat8_e4m3fnuz: [[fallthrough]];
    case DLDataTypeCode::kDLFloat8_e5m2: [[fallthrough]];
    case DLDataTypeCode::kDLFloat8_e5m2fnuz: [[fallthrough]];
    case DLDataTypeCode::kDLFloat8_e8m0fnu: [[fallthrough]];
    case DLDataTypeCode::kDLFloat6_e2m3fn: [[fallthrough]];
    case DLDataTypeCode::kDLFloat6_e3m2fn: [[fallthrough]];
    case DLDataTypeCode::kDLFloat4_e2m1fn: {
      throw TracedException<std::invalid_argument>{
        fmt::format("Conversion of DLPack type code {} to Legate not yet supported", code)};
    }
  }
  // NOLINTEND(readability-magic-numbers)
  LEGATE_ABORT("Unhandled DLPack type code ", dtype.code);
}

[[nodiscard]] bool is_tensor_contiguous(const DLTensor& tensor)
{
  // We compute the "farthest" index away from the origin of the tensor. If that index is 1
  // less than our volume, then it means the tensor is contiguous. If it had any "holes" (where
  // strides are larger than the shape), then the index would be larger than the volume.
  //
  // We cannot just early-out as soon as tensor.strides[i] > tensor.shape[i] because the array
  // may be in row-major order in which case the strides are always > tensor.shape[i].
  const auto tensor_strides =
    Span<const std::int64_t>{tensor.strides, tensor.strides + tensor.ndim};
  const auto tensor_shape   = Span<const std::int64_t>{tensor.shape, tensor.shape + tensor.ndim};
  std::int64_t farthest_idx = 0;
  std::int64_t volume       = 1;

  for (const auto [stride, shape] : zip_equal(tensor_strides, tensor_shape)) {
    farthest_idx += stride * (shape - 1);
    volume *= shape;
  }
  // Empty tensors (either actually empty: (), or a zero dimension: (1, 0, 2)) are always
  // contiguous.
  return volume == 0 || farthest_idx == volume - 1;
}

[[nodiscard]] legate::mapping::DimOrdering get_dim_ordering(const DLTensor& tensor)
{
  // From DLPack docs on the strides member:
  //
  // > strides of the tensor (in number of elements, not bytes) can be NULL, indicating tensor
  // > is compact and row-majored.
  if (!tensor.strides) {
    return legate::mapping::DimOrdering::c_order();
  }

  if (!is_tensor_contiguous(tensor)) {
    throw TracedException<std::invalid_argument>{
      "Conversion of non-contiguous strided tensors is not yet supported"};
  }

  const auto strides = Span<const std::int64_t>{tensor.strides, tensor.strides + tensor.ndim};

  // The order in which we check for row or column major is arbitrary, but we assume that the
  // vast majority of codes will give us row-major, so we check that first.
  if (std::is_sorted(strides.begin(), strides.end(), std::greater<>{})) {
    // Strides are ordered high-to-low, so row-major
    return legate::mapping::DimOrdering::c_order();
  }
  if (std::is_sorted(strides.begin(), strides.end())) {
    // Strides are ordered low-to-high, so column-major order
    return legate::mapping::DimOrdering::fortran_order();
  }
  // TODO(jfaibussowit)
  // We should probably support some kind of "strides" constructor here.
  throw TracedException<std::invalid_argument>{
    fmt::format("Cannot represent non monotonous {} strides as Legate dimension ordering",
                fmt::join(strides, ", "))};
}

[[nodiscard]] legate::ExternalAllocation make_external_alloc(const DLTensor& tensor,
                                                             bool read_only,
                                                             std::function<void(void*)> deleter)
{
  const auto size    = dl_tensor_size(tensor);
  auto* const ptr    = static_cast<std::byte*>(tensor.data) + tensor.byte_offset;
  const auto& device = tensor.device;
  const auto target  = to_store_target(device.device_type);

  switch (target) {
    case mapping::StoreTarget::SYSMEM: [[fallthrough]];
    case mapping::StoreTarget::SOCKETMEM:
      return legate::ExternalAllocation::create_sysmem(ptr, size, read_only, std::move(deleter));
    case mapping::StoreTarget::FBMEM:
      return legate::ExternalAllocation::create_fbmem(
        static_cast<std::uint32_t>(device.device_id), ptr, size, read_only, std::move(deleter));
    case mapping::StoreTarget::ZCMEM:
      return legate::ExternalAllocation::create_zcmem(ptr, size, read_only, std::move(deleter));
  }
  LEGATE_ABORT("Unhandled store target ", target);
}

[[nodiscard]] legate::LogicalStore from_dlpack(const DLTensor& tensor,
                                               bool read_only,
                                               std::function<void(void*)> deleter)
{
  if ((tensor.ndim < 0) || (tensor.ndim > LEGATE_MAX_DIM)) {
    throw TracedException<std::out_of_range>{fmt::format(
      "DLPack tensor dimension {} exceeds LEGATE_MAX_DIM (0, {}), recompile Legate with a "
      "larger maximum dimension to support this conversion",
      tensor.ndim,
      LEGATE_MAX_DIM)};
  }

  const auto shape     = dl_tensor_shape(tensor);
  const auto ty        = dl_tensor_type(tensor.dtype);
  const auto dim_order = get_dim_ordering(tensor);
  const auto alloc     = make_external_alloc(tensor, read_only, std::move(deleter));

  return legate::Runtime::get_runtime()->create_store(shape, ty, alloc, dim_order);
}

}  // namespace

legate::LogicalStore from_dlpack(DLManagedTensorVersioned** dlm_tensor)
{
  if (!dlm_tensor) {
    throw TracedException<std::invalid_argument>{
      "DLManagedTensorVersioned** argument must not be NULL"};
  }
  if (!*dlm_tensor) {
    throw TracedException<std::invalid_argument>{"DLManagedTensorVersioned* must not be NULL"};
  }
  // Hold in a unique_ptr to make sure the tensor is properly deleted even if any of the below
  // throw exceptions.
  auto uniq = std::unique_ptr<DLManagedTensorVersioned, std::function<void(void*)>>{
    *dlm_tensor, [tptr = *dlm_tensor](void*) {
      // We ignore the input argument here because this deleter function is also used as the
      // ExternalAllocation deleter. So for this unique_ptr the argument would be the original
      // DLManagedTensorVersioned, but for ExternalAllocation it would be
      // dlm_tensor->dl_tensor.data.
      if (tptr->deleter) {
        tptr->deleter(tptr);
      }
    }};

  if (const auto major = uniq->version.major; major != DLPACK_MAJOR_VERSION) {
    throw TracedException<std::runtime_error>{
      fmt::format("Unsupported major version for DLPack tensor {} (Legate only supports {}.x)",
                  major,
                  DLPACK_MAJOR_VERSION)};
  }

  auto ret =
    from_dlpack(uniq->dl_tensor, uniq->flags & DLPACK_FLAG_BITMASK_READ_ONLY, uniq.get_deleter());
  // Only now, after creating the store, can be we sure that Legate has fully taken control of
  // the tensor
  std::ignore = uniq.release();
  *dlm_tensor = nullptr;
  return ret;
}

legate::LogicalStore from_dlpack(DLManagedTensor** dlm_tensor)
{
  if (!dlm_tensor) {
    throw TracedException<std::invalid_argument>{"DLManagedTensor** argument must not be NULL"};
  }
  if (!*dlm_tensor) {
    throw TracedException<std::invalid_argument>{"DLManagedTensor* must not be NULL"};
  }
  // Hold in a unique_ptr to make sure the tensor is properly deleted even if any of the below
  // throw exceptions.
  auto uniq = std::unique_ptr<DLManagedTensor, std::function<void(void*)>>{
    *dlm_tensor, [tptr = *dlm_tensor](void*) {
      // We ignore the input argument here because this deleter function is also used as the
      // ExternalAllocation deleter. So for this unique_ptr the argument would be the original
      // DLManagedTensor, but for ExternalAllocation it would be dlm_tensor->dl_tensor.data.
      if (tptr->deleter) {
        tptr->deleter(tptr);
      }
    }};
  auto ret = from_dlpack(uniq->dl_tensor, /* read_only */ false, uniq.get_deleter());
  // Only now, after creating the store, can be we sure that Legate has fully taken control of
  // the tensor
  std::ignore = uniq.release();
  *dlm_tensor = nullptr;
  return ret;
}

}  // namespace legate::detail
