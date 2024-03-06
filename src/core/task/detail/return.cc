/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/task/detail/return.h"

#include "core/task/detail/returned_exception_common.h"
#include "core/utilities/machine.h"
#include "core/utilities/typedefs.h"
#if LegateDefined(LEGATE_USE_CUDA)
#include "core/cuda/cuda_help.h"
#include "core/cuda/stream_pool.h"
#endif

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

namespace legate::detail {

ReturnValues::ReturnValues(std::vector<ReturnValue>&& return_values)
  : return_values_{std::move(return_values)}
{
  switch (return_values_.size()) {
    case 0: break;
    case 1: buffer_size_ = return_values_.front().size(); break;
    default:
      // total number of values
      buffer_size_ += max_aligned_size_for_type<std::uint32_t>();
      for (auto& ret : return_values_) {
        // offset to scalar i, note we do not align here because these are packed immediately
        // after the total number (which is also a std::uint32_t), and therefore will never
        // need to be aligned up or down.
        buffer_size_ += sizeof(std::uint32_t);
        // size of value to pack
        buffer_size_ += ret.size();
      }
      break;
  }
}

void ReturnValues::legion_serialize(void* buffer) const
{
  // We pack N return values into the buffer in the following format:
  //
  // +--------+-----------+-----+------------+-------+-------+-------+-----
  // |   #    | offset to |     | offset to  | total | value | value | ...
  // | values | scalar 1  | ... | scalar N-1 | value |   1   |   2   |
  // |        |           |     |            | size  |       |       |
  // +--------+-----------+-----+------------+-------+-------+-------+-----
  //           <============ offsets ===============> <==== values =======>
  //
  // the size of value i is computed by offsets[i] - (i == 0 ? 0 : offsets[i-1])
  auto rem_cap = legion_buffer_size();

  // Special case with a single scalar
  if (return_values_.size() == 1) {
    auto& ret = return_values_.front();

    if (ret.is_device_value()) {
      LegateAssert(Processor::get_executing_processor().kind() == Processor::Kind::TOC_PROC);
      // TODO (jfaibussowit): expose cudaMemcpyAsync() as a stub instead
#if LegateDefined(LEGATE_USE_CUDA)
      CHECK_CUDA(cudaMemcpyAsync(buffer,
                                 ret.ptr(),
                                 ret.size(),
                                 cudaMemcpyDeviceToHost,
                                 cuda::StreamPool::get_stream_pool().get_stream()));
#endif
    } else {
      std::tie(buffer, rem_cap) =
        pack_buffer(buffer, rem_cap, ret.size(), static_cast<const char*>(ret.ptr()));
    }
    return;
  }

  std::tie(buffer, rem_cap) =
    pack_buffer(buffer, rem_cap, static_cast<std::uint32_t>(return_values_.size()));

  std::uint32_t offset = 0;
  for (auto&& ret : return_values_) {
    offset += ret.size();

    std::tie(buffer, rem_cap) = pack_buffer(buffer, rem_cap, offset);
  }

#if LegateDefined(LEGATE_USE_CUDA)
  if (Processor::get_executing_processor().kind() == Processor::Kind::TOC_PROC) {
    auto stream = cuda::StreamPool::get_stream_pool().get_stream();

    for (auto&& ret : return_values_) {
      const auto size = ret.size();

      if (ret.is_device_value()) {
        CHECK_CUDA(cudaMemcpyAsync(buffer, ret.ptr(), size, cudaMemcpyDeviceToHost, stream));
        buffer = static_cast<char*>(buffer) + size;
        rem_cap -= size;
      } else {
        std::tie(buffer, rem_cap) =
          pack_buffer(buffer, rem_cap, size, static_cast<const char*>(ret.ptr()));
      }
    }
  } else
#endif
  {
    for (auto&& ret : return_values_) {
      std::tie(buffer, rem_cap) =
        pack_buffer(buffer, rem_cap, ret.size(), static_cast<const char*>(ret.ptr()));
    }
  }
}

namespace {

[[nodiscard]] std::tuple<std::uint32_t, const std::uint32_t*, const char*>
extract_offsets_and_values(const void* buffer,
                           std::size_t rem_cap = std::numeric_limits<std::size_t>::max())
{
  std::uint32_t num_values{};
  std::tie(buffer, std::ignore) = unpack_buffer(buffer, rem_cap, &num_values);
  const auto offsets            = static_cast<const std::uint32_t*>(buffer);
  const auto values = reinterpret_cast<const char*>(offsets) + (sizeof(*offsets) * num_values);

  return {num_values, offsets, values};
}

}  // namespace

void ReturnValues::legion_deserialize(const void* buffer)
{
  const auto mem_kind                      = find_memory_kind_for_executing_processor();
  const auto [num_values, offsets, values] = extract_offsets_and_values(buffer);

  return_values_.reserve(num_values);
  for (std::uint32_t idx = 0, offset = 0; idx < num_values; ++idx) {
    const auto next_offset = offsets[idx];
    const auto size        = next_offset - offset;

    return_values_.push_back(ReturnValue::unpack(values + offset, size, mem_kind));
    offset = next_offset;
  }
}

/*static*/ ReturnValue ReturnValues::extract(const Legion::Future& future, std::uint32_t to_extract)
{
  const auto kind = find_memory_kind_for_executing_processor();
  const auto [_, offsets, values] =
    extract_offsets_and_values(future.get_buffer(kind), future.get_untyped_size());
  const auto next_offset = offsets[to_extract];
  const auto offset      = to_extract == 0 ? 0 : offsets[to_extract - 1];
  const auto size        = next_offset - offset;

  return ReturnValue::unpack(values + offset, size, kind);
}

void ReturnValues::finalize(Legion::Context legion_context) const
{
  if (return_values_.empty()) {
    Legion::Runtime::legion_task_postamble(legion_context);
    return;
  }
  if (return_values_.size() == 1) {
    return_values_.front().finalize(legion_context);
    return;
  }

#if LegateDefined(LEGATE_USE_CUDA)
  auto kind = Processor::get_executing_processor().kind();
  // FIXME: We don't currently have a good way to defer the return value packing on GPUs,
  //        as doing so would require the packing to be chained up with all preceding kernels,
  //        potentially launched with different streams, within the task. Until we find
  //        the right approach, we simply synchronize the device before proceeding.
  if (kind == Processor::TOC_PROC) {
    CHECK_CUDA(cudaDeviceSynchronize());
  }
#endif

  const std::size_t return_size = legion_buffer_size();
  auto return_buffer =
    Legion::UntypedDeferredValue{return_size, find_memory_kind_for_executing_processor()};
  const AccessorWO<std::int8_t, 1> acc{return_buffer, return_size, false};

  legion_serialize(acc.ptr(0));
  return_buffer.finalize(legion_context);
}

}  // namespace legate::detail
