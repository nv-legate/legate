/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/comm/coll_comm.h>
#include <legate/type/detail/types.h>
#include <legate/type/types.h>

#include <nccl.h>

namespace legate::detail {

[[nodiscard]] legate::comm::coll::CollDataType to_coll_data_type_(Type::Code type_code);

[[nodiscard]] legate::comm::coll::CollDataType to_coll_data_type_(std::size_t field_size);

[[nodiscard]] ncclDataType_t to_nccl_data_type(Type::Code type_code);

[[nodiscard]] ncclDataType_t to_nccl_data_type(std::size_t field_size);

[[nodiscard]] ncclRedOp_t to_nccl_red_op(ReductionOpKind op_kind);

}  // namespace legate::detail
