/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/copy_request.h>

#include <legate/comm/coll.h>
#include <legate/comm/coll_comm.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/traced_exception.h>

#include <nccl.h>
#include <stdexcept>

namespace legate::detail {

namespace {

[[nodiscard]] legate::comm::coll::CollDataType to_coll_data_type_(Type::Code type_code)
{
  switch (type_code) {
    case Type::Code::INT8: return legate::comm::coll::CollDataType::CollInt8;
    case Type::Code::INT32: return legate::comm::coll::CollDataType::CollInt;
    case Type::Code::INT64: return legate::comm::coll::CollDataType::CollInt64;
    case Type::Code::UINT8: return legate::comm::coll::CollDataType::CollUint8;
    case Type::Code::UINT32: return legate::comm::coll::CollDataType::CollUint32;
    case Type::Code::UINT64: return legate::comm::coll::CollDataType::CollUint64;
    case Type::Code::FLOAT32: return legate::comm::coll::CollDataType::CollFloat;
    case Type::Code::FLOAT64: return legate::comm::coll::CollDataType::CollDouble;

    case Type::Code::BOOL: [[fallthrough]];
    case Type::Code::INT16: [[fallthrough]];
    case Type::Code::UINT16: [[fallthrough]];
    case Type::Code::FLOAT16: [[fallthrough]];
    case Type::Code::COMPLEX64: [[fallthrough]];
    case Type::Code::COMPLEX128: [[fallthrough]];
    case Type::Code::NIL: [[fallthrough]];
    case Type::Code::BINARY: [[fallthrough]];
    case Type::Code::FIXED_ARRAY: [[fallthrough]];
    case Type::Code::STRUCT: [[fallthrough]];
    case Type::Code::STRING: [[fallthrough]];
    case Type::Code::LIST: {
      throw TracedException<std::runtime_error>{"Not supported yet"};
    }
  }

  LEGATE_UNREACHABLE();
  return legate::comm::coll::CollDataType::CollInt8;
}

[[nodiscard]] ncclDataType_t to_nccl_data_type(Type::Code type_code)
{
  switch (type_code) {
    case Type::Code::INT8: return ncclInt8;
    case Type::Code::INT32: return ncclInt32;
    case Type::Code::INT64: return ncclInt64;
    case Type::Code::UINT8: return ncclUint8;
    case Type::Code::UINT32: return ncclUint32;
    case Type::Code::UINT64: return ncclUint64;
    case Type::Code::FLOAT32: return ncclFloat32;
    case Type::Code::FLOAT64: return ncclFloat64;

    case Type::Code::BOOL: [[fallthrough]];
    case Type::Code::INT16: [[fallthrough]];
    case Type::Code::UINT16: [[fallthrough]];
    case Type::Code::FLOAT16: [[fallthrough]];
    case Type::Code::COMPLEX64: [[fallthrough]];
    case Type::Code::COMPLEX128: [[fallthrough]];
    case Type::Code::NIL: [[fallthrough]];
    case Type::Code::BINARY: [[fallthrough]];
    case Type::Code::FIXED_ARRAY: [[fallthrough]];
    case Type::Code::STRUCT: [[fallthrough]];
    case Type::Code::STRING: [[fallthrough]];
    case Type::Code::LIST: {
      throw TracedException<std::runtime_error>{"Not supported yet"};
    }
  }

  LEGATE_UNREACHABLE();
  return ncclInt8;
}

[[nodiscard]] ncclRedOp_t to_nccl_red_op(ReductionOpKind op_kind)
{
  switch (op_kind) {
    case ReductionOpKind::ADD: return ncclSum;
    case ReductionOpKind::MUL: return ncclProd;
    case ReductionOpKind::MAX: return ncclMax;
    case ReductionOpKind::MIN: return ncclMin;
    case ReductionOpKind::OR: [[fallthrough]];
    case ReductionOpKind::AND: [[fallthrough]];
    case ReductionOpKind::XOR: {
      throw TracedException<std::runtime_error>{"Not supported yet"};
    }
  }

  LEGATE_UNREACHABLE();
  return ncclSum;
}

}  // namespace

void ReductionRequest::perform(VariantCode variant_code) const
{
  if (variant_code == VariantCode::GPU) {
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

    // TODO(wonchanl): Error checking
    ncclAllReduce(source_->data(),
                  target_->data(),
                  source_->domain().get_volume(),
                  to_nccl_data_type(type_->code),
                  to_nccl_red_op(op_kind_),
                  Runtime::get_runtime().get_gpu_comm(),
                  Runtime::get_runtime().get_cuda_stream());

    // TODO(wonchanl): Hoist this logic to the task premable
    cuda::detail::get_cuda_driver_api()->stream_synchronize(
      Runtime::get_runtime().get_cuda_stream());
  } else {
    LEGATE_ASSERT(variant_code == VariantCode::CPU);

    collAllreduce(source_->data(),
                  target_->data(),
                  source_->domain().get_volume(),
                  to_coll_data_type_(type_->code),
                  op_kind_,
                  Runtime::get_runtime().get_cpu_comm());
  }
}

}  // namespace legate::detail
