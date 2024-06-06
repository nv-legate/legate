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

#include "core/cuda/cuda.h"
#include "core/cuda/stream_pool.h"
#include "core/partitioning/detail/partitioning_tasks.h"
#include "core/task/task_context.h"
#include "core/utilities/detail/unravel.h"

#include "core/utilities/detail/cuda_reduction_buffer.cuh"

#include <cstring>

namespace legate::detail {

namespace {

__device__ __forceinline__ constexpr std::size_t round_div(std::size_t x, std::size_t d)
{
  return (x + d - 1) / d;
}

// The following is copied from cuNumeric and modified
template <typename T>
__device__ __forceinline__ T shuffle(unsigned mask, T var, int laneMask, int width)
{
  int array[round_div(sizeof(T), sizeof(int))];
  std::memcpy(array, &var, sizeof(T));
  for (int& value : array) {
    value = __shfl_xor_sync(mask, value, laneMask, width);
  }
  std::memcpy(&var, array, sizeof(T));
  return var;
}

template <typename RED_LOW, typename RED_HIGH, std::int32_t NDIM>
__device__ __forceinline__ void block_reduce(CUDAReductionBuffer<RED_LOW> out_low,
                                             CUDAReductionBuffer<RED_HIGH> out_high,
                                             Point<NDIM> local_low,
                                             Point<NDIM> local_high)
{
  __shared__ Point<NDIM> block_low[LEGATE_THREADS_PER_BLOCK / LEGATE_WARP_SIZE];
  __shared__ Point<NDIM> block_high[LEGATE_THREADS_PER_BLOCK / LEGATE_WARP_SIZE];

  // Reduce across the warp
  int lane_id = threadIdx.x & (LEGATE_WARP_SIZE - 1);
  int warp_id = threadIdx.x >> 5;
  for (int i = 16; i >= 1; i /= 2) {
    auto shuffle_low  = shuffle(0xffffffff, local_low, i, LEGATE_WARP_SIZE);
    auto shuffle_high = shuffle(0xffffffff, local_high, i, LEGATE_WARP_SIZE);
    RED_LOW::template fold<true /*exclusive*/>(local_low, shuffle_low);
    RED_HIGH::template fold<true /*exclusive*/>(local_high, shuffle_high);
  }
  // Write warp values into shared memory
  if ((lane_id == 0) && (warp_id > 0)) {
    block_low[warp_id]  = local_low;
    block_high[warp_id] = local_high;
  }
  __syncthreads();

  // Output reduction
  if (threadIdx.x == 0) {
    for (int i = 1; i < (LEGATE_THREADS_PER_BLOCK / LEGATE_WARP_SIZE); i++) {
      RED_LOW::template fold<true /*exclusive*/>(local_low, block_low[i]);
      RED_HIGH::template fold<true /*exclusive*/>(local_high, block_high[i]);
    }
    out_low.reduce<false /*EXCLUSIVE*/>(local_low);
    out_high.reduce<false /*EXCLUSIVE*/>(local_high);

    // Make sure the result is visible externally
    __threadfence_system();
  }
}

template <bool RECT,
          std::int32_t STORE_NDIM,
          std::int32_t POINT_NDIM,
          typename RED_LOW,
          typename RED_HIGH,
          typename In>
__global__ void __launch_bounds__(LEGATE_THREADS_PER_BLOCK, LEGATE_MIN_CTAS_PER_SM)
  find_bounding_box_kernel(Unravel<STORE_NDIM> unravel,
                           std::size_t num_iters,
                           CUDAReductionBuffer<RED_LOW> out_low,
                           CUDAReductionBuffer<RED_HIGH> out_high,
                           In in,
                           Point<POINT_NDIM> identity_low,
                           Point<POINT_NDIM> identity_high)
{
  auto local_low  = identity_low;
  auto local_high = identity_high;

  for (std::size_t iter = 0; iter < num_iters; ++iter) {
    auto index = (iter * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < unravel.volume()) {
      auto p = unravel(index);
      if constexpr (RECT) {
        const auto& rect = in[p];
        RED_LOW::template fold<true /*EXCLUSIVE*/>(local_low, rect.lo);
        RED_HIGH::template fold<true /*EXCLUSIVE*/>(local_high, rect.hi);
      } else {
        const auto& point = in[p];
        RED_LOW::template fold<true /*EXCLUSIVE*/>(local_low, point);
        RED_HIGH::template fold<true /*EXCLUSIVE*/>(local_high, point);
      }
    }
  }
  // Every thread in the thread block must participate in the exchange to get correct results
  block_reduce(out_low, out_high, local_low, local_high);
}

template <bool RECT, std::int32_t STORE_NDIM, typename RED_LOW, typename RED_HIGH, typename In>
__global__ void __launch_bounds__(LEGATE_THREADS_PER_BLOCK, LEGATE_MIN_CTAS_PER_SM)
  find_bounding_box_sorted_kernel(Unravel<STORE_NDIM> unravel,
                                  CUDAReductionBuffer<RED_LOW> out_low,
                                  CUDAReductionBuffer<RED_HIGH> out_high,
                                  In in)
{
  if constexpr (RECT) {
    const auto& first_rect = in[unravel(0)];
    const auto& last_rect  = in[unravel(unravel.volume() - 1)];
    out_low.reduce<true /*EXCLUSIVE*/>(first_rect.lo);
    out_low.reduce<true /*EXCLUSIVE*/>(last_rect.lo);
    out_high.reduce<true /*EXCLUSIVE*/>(first_rect.hi);
    out_high.reduce<true /*EXCLUSIVE*/>(last_rect.hi);
  } else {
    const auto& first_point = in[unravel(0)];
    const auto& last_point  = in[unravel(unravel.volume() - 1)];
    out_low.reduce<true /*EXCLUSIVE*/>(first_point);
    out_low.reduce<true /*EXCLUSIVE*/>(last_point);
    out_high.reduce<true /*EXCLUSIVE*/>(first_point);
    out_high.reduce<true /*EXCLUSIVE*/>(last_point);
  }
}

template <typename Out, std::int32_t NDIM>
__global__ void __launch_bounds__(1, 1)
  copy_output(Out out,
              CUDAReductionBuffer<ElementWiseMin<NDIM>> in_low,
              CUDAReductionBuffer<ElementWiseMax<NDIM>> in_high)
{
  out[0] = Rect<NDIM>{in_low.read(), in_high.read()};
}

template <bool RECT>
struct FindBoundingBoxFn {
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::PhysicalStore& input, const legate::PhysicalStore& output)
  {
    auto shape   = input.shape<STORE_NDIM>();
    auto out_acc = output.write_accessor<Domain, 1>();

    auto stream = cuda::StreamPool::get_stream_pool().get_stream();

    const auto unravel = Unravel<STORE_NDIM>{shape};

    auto result_low  = CUDAReductionBuffer<ElementWiseMin<POINT_NDIM>>{stream};
    auto result_high = CUDAReductionBuffer<ElementWiseMax<POINT_NDIM>>{stream};

    std::size_t blocks     = round_div(unravel.volume(), LEGATE_THREADS_PER_BLOCK);
    std::size_t shmem_size = LEGATE_THREADS_PER_BLOCK / 32 * sizeof(Point<POINT_NDIM>) * 2;
    std::size_t num_iters  = round_div(blocks, LEGATE_MAX_REDUCTION_CTAS);

    if (!unravel.empty()) {
      if constexpr (RECT) {
        auto in_acc = input.read_accessor<Rect<POINT_NDIM>, STORE_NDIM>(shape);
        find_bounding_box_kernel<true>
          <<<LEGATE_MAX_REDUCTION_CTAS, LEGATE_THREADS_PER_BLOCK, shmem_size, stream>>>(
            unravel,
            num_iters,
            result_low,
            result_high,
            in_acc,
            ElementWiseMin<POINT_NDIM>::identity,
            ElementWiseMax<POINT_NDIM>::identity);
      } else {
        auto in_acc = input.read_accessor<Point<POINT_NDIM>, STORE_NDIM>(shape);
        find_bounding_box_kernel<false>
          <<<LEGATE_MAX_REDUCTION_CTAS, LEGATE_THREADS_PER_BLOCK, shmem_size, stream>>>(
            unravel,
            num_iters,
            result_low,
            result_high,
            in_acc,
            ElementWiseMin<POINT_NDIM>::identity,
            ElementWiseMax<POINT_NDIM>::identity);
      }
    }
    LEGATE_CHECK_CUDA_STREAM(stream);

    copy_output<<<1, 1, 0, stream>>>(out_acc, result_low, result_high);
    LEGATE_CHECK_CUDA_STREAM(stream);
  }
};

template <bool RECT>
struct FindBoundingBoxSortedFn {
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::PhysicalStore& input, const legate::PhysicalStore& output)
  {
    auto shape   = input.shape<STORE_NDIM>();
    auto out_acc = output.write_accessor<Domain, 1>();

    auto stream = cuda::StreamPool::get_stream_pool().get_stream();

    const auto unravel = Unravel<STORE_NDIM>{shape};

    auto result_low  = CUDAReductionBuffer<ElementWiseMin<POINT_NDIM>>{stream};
    auto result_high = CUDAReductionBuffer<ElementWiseMax<POINT_NDIM>>{stream};

    if (!unravel.empty()) {
      if constexpr (RECT) {
        auto in_acc = input.read_accessor<Rect<POINT_NDIM>, STORE_NDIM>(shape);
        find_bounding_box_sorted_kernel<true>
          <<<1, 1, 0, stream>>>(unravel, result_low, result_high, in_acc);
      } else {
        auto in_acc = input.read_accessor<Point<POINT_NDIM>, STORE_NDIM>(shape);
        find_bounding_box_sorted_kernel<false>
          <<<1, 1, 0, stream>>>(unravel, result_low, result_high, in_acc);
      }
    }
    LEGATE_CHECK_CUDA_STREAM(stream);

    copy_output<<<1, 1, 0, stream>>>(out_acc, result_low, result_high);
    LEGATE_CHECK_CUDA_STREAM(stream);
  }
};

}  // namespace

/*static*/ void FindBoundingBox::gpu_variant(legate::TaskContext context)
{
  auto input  = context.input(0).data();
  auto output = context.output(0).data();

  auto type = input.type();

  if (legate::is_rect_type(type)) {
    legate::double_dispatch(
      legate::ndim_rect_type(type), input.dim(), FindBoundingBoxFn<true>{}, input, output);
  } else {
    legate::double_dispatch(
      legate::ndim_point_type(type), input.dim(), FindBoundingBoxFn<false>{}, input, output);
  }
}

/*static*/ void FindBoundingBoxSorted::gpu_variant(legate::TaskContext context)
{
  auto input  = context.input(0).data();
  auto output = context.output(0).data();

  auto type = input.type();

  if (legate::is_rect_type(type)) {
    legate::double_dispatch(
      legate::ndim_rect_type(type), input.dim(), FindBoundingBoxSortedFn<true>{}, input, output);
  } else {
    legate::double_dispatch(
      legate::ndim_point_type(type), input.dim(), FindBoundingBoxSortedFn<false>{}, input, output);
  }
}

}  // namespace legate::detail
