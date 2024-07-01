/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/task/detail/task_return.h"

#include "core/cuda/cuda.h"
#include "core/cuda/stream_pool.h"
#include "core/runtime/detail/runtime.h"
#include "core/task/detail/returned_exception_common.h"
#include "core/utilities/detail/zip.h"
#include "core/utilities/machine.h"
#include "core/utilities/typedefs.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <tuple>
#include <vector>

namespace legate::detail {

TaskReturn::TaskReturn(std::vector<ReturnValue>&& return_values)
  : return_values_{std::move(return_values)}, layout_{return_values_}
{
}

void TaskReturn::pack(void* buffer) const
{
  auto out_ptr = static_cast<std::int8_t*>(buffer);

  // Special case with a single scalar
  LEGATE_ASSERT(return_values_.size() > 1);

  if (detail::Runtime::get_runtime()->get_executing_processor().kind() ==
      Processor::Kind::TOC_PROC) {
    auto stream = cuda::StreamPool::get_stream_pool().get_stream();

    for (auto&& [ret, offset] : zip_equal(return_values_, layout_)) {
      if (ret.is_device_value()) {
        LEGATE_CHECK_CUDA(
          cudaMemcpyAsync(out_ptr + offset, ret.ptr(), ret.size(), cudaMemcpyDeviceToHost, stream));
      } else {
        std::memcpy(out_ptr + offset, ret.ptr(), ret.size());
      }
    }
  } else {
    for (auto&& [ret, offset] : zip_equal(return_values_, layout_)) {
      std::memcpy(out_ptr + offset, ret.ptr(), ret.size());
    }
  }
}

void TaskReturn::finalize(Legion::Context legion_context) const
{
  if (return_values_.empty()) {
    Legion::Runtime::legion_task_postamble(legion_context);
    return;
  }
  if (return_values_.size() == 1) {
    return_values_.front().finalize(legion_context);
    return;
  }

  auto kind = detail::Runtime::get_runtime()->get_executing_processor().kind();
  // FIXME: We don't currently have a good way to defer the return value packing on GPUs,
  //        as doing so would require the packing to be chained up with all preceding kernels,
  //        potentially launched with different streams, within the task. Until we find
  //        the right approach, we simply synchronize the device before proceeding.
  if (kind == Processor::TOC_PROC) {
    LEGATE_CHECK_CUDA(cudaDeviceSynchronize());
  }

  auto return_buffer =
    Legion::UntypedDeferredValue{buffer_size(), find_memory_kind_for_executing_processor()};
  const AccessorWO<std::int8_t, 1> acc{return_buffer, buffer_size(), false};

  pack(acc.ptr(0));
  return_buffer.finalize(legion_context);
}

}  // namespace legate::detail
