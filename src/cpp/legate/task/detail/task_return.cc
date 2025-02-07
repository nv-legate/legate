/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/task/detail/task_return.h>

#include <legate/cuda/cuda.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/detail/returned_exception_common.h>
#include <legate/utilities/detail/align.h>
#include <legate/utilities/detail/zip.h>
#include <legate/utilities/machine.h>
#include <legate/utilities/typedefs.h>

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

  if (auto* runtime = detail::Runtime::get_runtime();
      runtime->get_executing_processor().kind() == Processor::Kind::TOC_PROC) {
    auto stream = runtime->get_cuda_stream();

    for (auto&& [ret, offset] : zip_equal(return_values_, layout_)) {
      if (ret.is_device_value()) {
        runtime->get_cuda_driver_api()->mem_cpy_async(
          out_ptr + offset, ret.ptr(), ret.size(), stream);
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

void TaskReturn::finalize(Legion::Context legion_context, bool skip_device_ctx_sync) const
{
  if (return_values_.empty()) {
    Legion::Runtime::legion_task_postamble(legion_context);
    return;
  }
  if (return_values_.size() == 1) {
    return_values_.front().finalize(legion_context);
    return;
  }

  if (!skip_device_ctx_sync) {
    auto* runtime   = detail::Runtime::get_runtime();
    const auto kind = runtime->get_executing_processor().kind();
    // FIXME: We don't currently have a good way to defer the return value packing on GPUs,
    //        as doing so would require the packing to be chained up with all preceding kernels,
    //        potentially launched with different streams, within the task. Until we find
    //        the right approach, we simply synchronize the device before proceeding.
    if (kind == Processor::TOC_PROC) {
      runtime->get_cuda_driver_api()->ctx_synchronize();
    }
  }

  // We don't have direct control on the alignment of future instances. So, when Legion is about to
  // make a copy of a future of some unusual size (say 27) due to multiplexing here, it may use a
  // wrong alignment for the first element packed in the instance, which can lead to misaligned
  // accesses in CUDA kernels. To prevent that from happening, here we align the size to the 16-byte
  // boundary and set the alignment so there'd be no room for misinterpretation.
  const auto aligned_size = round_up_to_multiple(buffer_size(), ALIGNMENT);
  auto return_buffer      = Legion::UntypedDeferredValue{aligned_size,
                                                    find_memory_kind_for_executing_processor(),
                                                    nullptr /*initial_value*/,
                                                    ALIGNMENT /*alignment*/};
  const AccessorWO<std::int8_t, 1> acc{return_buffer, buffer_size(), false};

  pack(acc.ptr(0));
  return_buffer.finalize(legion_context);
}

}  // namespace legate::detail
