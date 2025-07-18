/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/dlpack/common.h>

#include <legate/mapping/mapping.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <climits>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace fmt {

format_context::iterator formatter<DLDeviceType>::format(DLDeviceType dtype,
                                                         format_context& ctx) const
{
#define LEGATE_DEVICE_TYPE_CASE(name) \
  case DLDeviceType::kDL##name: return formatter<string_view>::format(#name, ctx)

  switch (dtype) {
    LEGATE_DEVICE_TYPE_CASE(CPU);
    LEGATE_DEVICE_TYPE_CASE(CUDA);
    LEGATE_DEVICE_TYPE_CASE(CUDAHost);
    LEGATE_DEVICE_TYPE_CASE(OpenCL);
    LEGATE_DEVICE_TYPE_CASE(Vulkan);
    LEGATE_DEVICE_TYPE_CASE(Metal);
    LEGATE_DEVICE_TYPE_CASE(VPI);
    LEGATE_DEVICE_TYPE_CASE(ROCM);
    LEGATE_DEVICE_TYPE_CASE(ROCMHost);
    LEGATE_DEVICE_TYPE_CASE(ExtDev);
    LEGATE_DEVICE_TYPE_CASE(CUDAManaged);
    LEGATE_DEVICE_TYPE_CASE(OneAPI);
    LEGATE_DEVICE_TYPE_CASE(WebGPU);
    LEGATE_DEVICE_TYPE_CASE(Hexagon);
    LEGATE_DEVICE_TYPE_CASE(MAIA);
  }
#undef LEGATE_DEVICE_TYPE_CASE

  return formatter<string_view>::format("Unknown DLPack device type", ctx);
}

format_context::iterator formatter<DLDataTypeCode>::format(DLDataTypeCode code,
                                                           format_context& ctx) const
{
#define LEGATE_DATA_TYPE_CODE_CASE(name) \
  case DLDataTypeCode::kDL##name: return formatter<string_view>::format(#name, ctx)

  switch (code) {
    LEGATE_DATA_TYPE_CODE_CASE(Int);
    LEGATE_DATA_TYPE_CODE_CASE(UInt);
    LEGATE_DATA_TYPE_CODE_CASE(Float);
    LEGATE_DATA_TYPE_CODE_CASE(Bfloat);
    LEGATE_DATA_TYPE_CODE_CASE(OpaqueHandle);
    LEGATE_DATA_TYPE_CODE_CASE(Complex);
    LEGATE_DATA_TYPE_CODE_CASE(Bool);
    LEGATE_DATA_TYPE_CODE_CASE(Float8_e3m4);
    LEGATE_DATA_TYPE_CODE_CASE(Float8_e4m3);
    LEGATE_DATA_TYPE_CODE_CASE(Float8_e4m3b11fnuz);
    LEGATE_DATA_TYPE_CODE_CASE(Float8_e4m3fn);
    LEGATE_DATA_TYPE_CODE_CASE(Float8_e4m3fnuz);
    LEGATE_DATA_TYPE_CODE_CASE(Float8_e5m2);
    LEGATE_DATA_TYPE_CODE_CASE(Float8_e5m2fnuz);
    LEGATE_DATA_TYPE_CODE_CASE(Float8_e8m0fnu);
    LEGATE_DATA_TYPE_CODE_CASE(Float6_e2m3fn);
    LEGATE_DATA_TYPE_CODE_CASE(Float6_e3m2fn);
    LEGATE_DATA_TYPE_CODE_CASE(Float4_e2m1fn);
  }

#undef LEGATE_DATA_TYPE_CODE_CASE

  return formatter<string_view>::format("Unknown DLPack data type ", ctx);
}

}  // namespace fmt

namespace legate::detail {

mapping::StoreTarget to_store_target(DLDeviceType dtype)
{
  switch (dtype) {
    case DLDeviceType::kDLCPU: return mapping::StoreTarget::SYSMEM;
    case DLDeviceType::kDLCUDAManaged:
      [[fallthrough]];
      // TODO(jfaibussowit):
      // Currently, managed memory is handled the same as if it were on the device, which could
      // technically be the less performance choice, if the allocation happens to be on host
      // memory at the moment. But we assume that if the user is giving us managed memory, then
      // surely they will want to use it on the device.
    case DLDeviceType::kDLCUDA: return mapping::StoreTarget::FBMEM;
    case DLDeviceType::kDLCUDAHost: return mapping::StoreTarget::ZCMEM;
    case DLDeviceType::kDLOpenCL: [[fallthrough]];
    case DLDeviceType::kDLVulkan: [[fallthrough]];
    case DLDeviceType::kDLMetal: [[fallthrough]];
    case DLDeviceType::kDLVPI: [[fallthrough]];
    case DLDeviceType::kDLROCM: [[fallthrough]];
    case DLDeviceType::kDLROCMHost: [[fallthrough]];
    case DLDeviceType::kDLExtDev: [[fallthrough]];
    case DLDeviceType::kDLOneAPI: [[fallthrough]];
    case DLDeviceType::kDLWebGPU: [[fallthrough]];
    case DLDeviceType::kDLHexagon: [[fallthrough]];
    case DLDeviceType::kDLMAIA:
      throw TracedException<std::invalid_argument>{fmt::format(
        "Conversion from DLPack device type {} to Legate store is not yet supported.", dtype)};
  }
  LEGATE_ABORT("Unhandled device type ", dtype);
}

DLDataTypeCode to_dlpack_type(legate::Type::Code code)
{
  switch (code) {
    case legate::Type::Code::BOOL: return DLDataTypeCode::kDLBool;
    case legate::Type::Code::INT8: [[fallthrough]];
    case legate::Type::Code::INT16: [[fallthrough]];
    case legate::Type::Code::INT32: [[fallthrough]];
    case legate::Type::Code::INT64: return DLDataTypeCode::kDLInt;
    case legate::Type::Code::UINT8: [[fallthrough]];
    case legate::Type::Code::UINT16: [[fallthrough]];
    case legate::Type::Code::UINT32: [[fallthrough]];
    case legate::Type::Code::UINT64: return DLDataTypeCode::kDLUInt;
    case legate::Type::Code::FLOAT16: return DLDataTypeCode::kDLBfloat;
    case legate::Type::Code::FLOAT32: [[fallthrough]];
    case legate::Type::Code::FLOAT64: return DLDataTypeCode::kDLFloat;
    case legate::Type::Code::COMPLEX64: [[fallthrough]];
    case legate::Type::Code::COMPLEX128: return DLDataTypeCode::kDLComplex;
    case legate::Type::Code::BINARY: return DLDataTypeCode::kDLOpaqueHandle;
    case legate::Type::Code::NIL: [[fallthrough]];
    case legate::Type::Code::FIXED_ARRAY: [[fallthrough]];
    case legate::Type::Code::STRUCT: [[fallthrough]];
    case legate::Type::Code::STRING: [[fallthrough]];
    case legate::Type::Code::LIST: {
      throw TracedException<std::invalid_argument>{
        fmt::format("Conversion of {} type to DLPack is not yet supported", fmt::underlying(code))};
    }
  }
  LEGATE_ABORT("Unhandled type code ", code);
}

std::size_t dl_tensor_size(const DLTensor& tensor)
{
  // Size in elements
  std::size_t size =
    array_volume(Span<const std::int64_t>{tensor.shape, tensor.shape + tensor.ndim});

  size *= (tensor.dtype.bits * tensor.dtype.lanes + (CHAR_BIT - 1)) / CHAR_BIT;
  return size;
}

}  // namespace legate::detail
