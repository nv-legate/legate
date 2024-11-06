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

#include "legate/cuda/cuda.h"
#include "legate/cuda/detail/cuda_driver_api.h"
#include "legate/data/detail/array_tasks.h"
#include "legate/runtime/detail/runtime.h"
#include "legate/task/task_context.h"

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <legate/generated/fatbin/fixup_ranges_fatbin.h>
#include <legate/generated/fatbin/offsets_to_ranges_fatbin.h>
#include <legate/generated/fatbin/ranges_to_offsets_fatbin.h>
#else
namespace {

// NOLINTBEGIN
constexpr const void* fixup_ranges_fatbin      = nullptr;
constexpr const void* offsets_to_ranges_fatbin = nullptr;
constexpr const void* ranges_to_offsets_fatbin = nullptr;
// NOLINTEND

}  // namespace
#endif

namespace legate::detail {

/*static*/ void FixupRanges::gpu_variant(legate::TaskContext context)
{
  if (context.get_task_index()[0] == 0) {
    return;
  }

  const auto num_outputs = context.num_outputs();
  const auto stream      = context.get_task_stream();
  auto* runtime          = Runtime::get_runtime();
  const auto* api        = runtime->get_cuda_driver_api();
  const auto kern        = runtime->get_cuda_module_manager()->load_kernel_from_fatbin(
    fixup_ranges_fatbin, "legate_fixup_ranges_kernel");

  // TODO(wonchanl): We need to extend this to nested cases
  for (std::uint32_t i = 0; i < num_outputs; ++i) {
    const auto list_arr = context.output(i).as_list_array();
    const auto desc     = list_arr.descriptor();
    auto desc_shape     = desc.shape<1>();
    if (desc_shape.empty()) {
      continue;
    }

    auto vardata_lo       = list_arr.vardata().shape<1>().lo;
    auto desc_acc         = desc.data().read_write_accessor<Rect<1>, 1>();
    auto desc_volume      = desc_shape.volume();
    const auto num_blocks = (desc_volume + LEGATE_THREADS_PER_BLOCK - 1) / LEGATE_THREADS_PER_BLOCK;
    void* kernel_params[] = {&desc_volume, &desc_shape.lo, &vardata_lo, &desc_acc};

    LEGATE_CHECK_CUDRIVER(api->launch_kernel(
      kern, {num_blocks}, {LEGATE_THREADS_PER_BLOCK}, 0, stream, kernel_params, nullptr));
  }
}

/*static*/ void OffsetsToRanges::gpu_variant(legate::TaskContext context)
{
  auto offsets = context.input(0).data();
  auto vardata = context.input(1).data();
  auto ranges  = context.output(0).data();

  auto offsets_shape = offsets.shape<1>();
  LEGATE_CHECK(offsets_shape == ranges.shape<1>());

  if (offsets_shape.empty()) {
    return;
  }

  const auto stream = context.get_task_stream();
  auto* runtime     = Runtime::get_runtime();
  const auto* api   = runtime->get_cuda_driver_api();
  const auto kern   = runtime->get_cuda_module_manager()->load_kernel_from_fatbin(
    offsets_to_ranges_fatbin, "legate_offsets_to_ranges_kernel");

  auto vardata_shape  = vardata.shape<1>();
  auto offsets_acc    = offsets.read_accessor<std::int32_t, 1>();
  auto ranges_acc     = ranges.write_accessor<Rect<1>, 1>();
  auto offsets_volume = offsets_shape.volume();
  auto vardata_volume = vardata_shape.volume();

  const auto num_blocks =
    (offsets_volume + LEGATE_THREADS_PER_BLOCK - 1) / LEGATE_THREADS_PER_BLOCK;

  void* kernel_params[] = {&offsets_volume,
                           &vardata_volume,
                           &offsets_shape.lo,
                           &vardata_shape.lo,
                           &ranges_acc,
                           &offsets_acc};

  LEGATE_CHECK_CUDRIVER(api->launch_kernel(
    kern, {num_blocks}, {LEGATE_THREADS_PER_BLOCK}, 0, stream, kernel_params, nullptr));
}

/*static*/ void RangesToOffsets::gpu_variant(legate::TaskContext context)
{
  auto ranges  = context.input(0).data();
  auto offsets = context.output(0).data();

  auto ranges_shape = ranges.shape<1>();
  LEGATE_CHECK(ranges_shape == offsets.shape<1>());

  if (ranges_shape.empty()) {
    return;
  }

  const auto stream = context.get_task_stream();
  auto* runtime     = Runtime::get_runtime();
  const auto* api   = runtime->get_cuda_driver_api();
  const auto kern   = runtime->get_cuda_module_manager()->load_kernel_from_fatbin(
    ranges_to_offsets_fatbin, "legate_ranges_to_offsets_kernel");

  auto ranges_acc       = ranges.read_accessor<Rect<1>, 1>();
  auto offsets_acc      = offsets.write_accessor<std::int32_t, 1>();
  auto ranges_volume    = ranges_shape.volume();
  const auto num_blocks = (ranges_volume + LEGATE_THREADS_PER_BLOCK - 1) / LEGATE_THREADS_PER_BLOCK;

  void* kernel_params[] = {&ranges_volume, &ranges_shape.lo, &offsets_acc, &ranges_acc};

  LEGATE_CHECK_CUDRIVER(api->launch_kernel(
    kern, {num_blocks}, {LEGATE_THREADS_PER_BLOCK}, 0, stream, kernel_params, nullptr));
}

}  // namespace legate::detail
