/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/copy_request.h>

#include <legate/comm/coll.h>
#include <legate/comm/coll_comm.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/traced_exception.h>

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

}  // namespace

void ReductionRequest::perform(VariantCode variant_code) const
{
  if (variant_code != VariantCode::CPU) {
    throw TracedException<std::runtime_error>{"Not supported yet"};
  }

  collAllreduce(source_->data(),
                target_->data(),
                /*count=*/1,
                to_coll_data_type_(type_->code),
                op_kind_,
                Runtime::get_runtime().get_cpu_comm());
}

}  // namespace legate::detail
