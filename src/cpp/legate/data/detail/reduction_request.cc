/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/reduction_request.h>

#include <legate/comm/coll.h>
#include <legate/comm/coll_comm.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/detail/copy_util.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/traced_exception.h>

#include <nccl.h>
#include <stdexcept>

namespace legate::detail {

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
