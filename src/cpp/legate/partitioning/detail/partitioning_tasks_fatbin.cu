/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partitioning_tasks.h>
#include <legate/utilities/detail/cuda_reduction_buffer.cuh>
#include <legate/utilities/detail/unravel.h>
#include <legate/utilities/typedefs.h>

#include <legion/legion_config.h>  // LEGION_FOREACH_N

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace detail {

namespace {

template <std::int32_t NDIM>
__device__ void copy_output(
  legate::AccessorWO<legate::Domain, 1> out,
  legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMin<NDIM>> in_low,
  legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMax<NDIM>> in_high)
{
  out[0] = legate::Rect<NDIM>{in_low.read(), in_high.read()};
}

}  // namespace

}  // namespace detail

#define COPY_OUTPUT_TPL_INST(NDIM)                                                     \
  extern "C" __global__ void __launch_bounds__(1, 1) legate_copy_output_##NDIM(        \
    legate::AccessorWO<legate::Domain, 1> out,                                         \
    legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMin<NDIM>> in_low,  \
    legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMax<NDIM>> in_high) \
  {                                                                                    \
    detail::copy_output<NDIM>(out, in_low, in_high);                                   \
  }

LEGION_FOREACH_N(COPY_OUTPUT_TPL_INST)

#undef COPY_OUTPUT_TPL_INST

// ==========================================================================================

namespace detail {

namespace {

template <std::int32_t STORE_NDIM, std::int32_t POINT_NDIM, typename InAcc>
__device__ void find_bounding_box_sorted_kernel(
  legate::detail::Unravel<STORE_NDIM> unravel,
  legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMin<POINT_NDIM>> out_low,
  legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMax<POINT_NDIM>> out_high,
  legate::AccessorRO<InAcc, STORE_NDIM> in)
{
  const auto& first = in[unravel(0)];
  const auto& last  = in[unravel(unravel.volume() - 1)];

  if constexpr (std::is_same_v<InAcc, legate::Rect<POINT_NDIM>>) {
    out_low.template reduce<true /*EXCLUSIVE*/>(first.lo);
    out_low.template reduce<true /*EXCLUSIVE*/>(last.lo);
    out_high.template reduce<true /*EXCLUSIVE*/>(first.hi);
    out_high.template reduce<true /*EXCLUSIVE*/>(last.hi);
  } else {
    out_low.template reduce<true /*EXCLUSIVE*/>(first);
    out_low.template reduce<true /*EXCLUSIVE*/>(last);
    out_high.template reduce<true /*EXCLUSIVE*/>(first);
    out_high.template reduce<true /*EXCLUSIVE*/>(last);
  }
}

}  // namespace

}  // namespace detail

#define FIND_BBOX_SORTED_TPL_INST(STORE_NDIM, POINT_NDIM)                                        \
  extern "C" __global__ void __launch_bounds__(LEGATE_THREADS_PER_BLOCK, LEGATE_MIN_CTAS_PER_SM) \
    legate_find_bounding_box_sorted_kernel_rect_##STORE_NDIM##_##POINT_NDIM(                     \
      legate::detail::Unravel<STORE_NDIM> unravel,                                               \
      legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMin<POINT_NDIM>> out_low,   \
      legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMax<POINT_NDIM>> out_high,  \
      legate::AccessorRO<legate::Rect<POINT_NDIM>, STORE_NDIM> in)                               \
  {                                                                                              \
    detail::find_bounding_box_sorted_kernel(unravel, out_low, out_high, in);                     \
  }                                                                                              \
                                                                                                 \
  extern "C" __global__ void __launch_bounds__(LEGATE_THREADS_PER_BLOCK, LEGATE_MIN_CTAS_PER_SM) \
    legate_find_bounding_box_sorted_kernel_point_##STORE_NDIM##_##POINT_NDIM(                    \
      legate::detail::Unravel<STORE_NDIM> unravel,                                               \
      legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMin<POINT_NDIM>> out_low,   \
      legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMax<POINT_NDIM>> out_high,  \
      legate::AccessorRO<legate::Point<POINT_NDIM>, STORE_NDIM> in)                              \
  {                                                                                              \
    detail::find_bounding_box_sorted_kernel(unravel, out_low, out_high, in);                     \
  }

LEGION_FOREACH_NN(FIND_BBOX_SORTED_TPL_INST)

#undef FIND_BBOX_SORTED_TPL_INST

// ==========================================================================================

namespace detail {

namespace {

__device__ __forceinline__ constexpr std::size_t round_div(std::size_t x, std::size_t d)
{
  return (x + d - 1) / d;
}

// The following is copied from cuPyNumeric and modified
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
__device__ __forceinline__ void block_reduce(legate::detail::CUDAReductionBuffer<RED_LOW> out_low,
                                             legate::detail::CUDAReductionBuffer<RED_HIGH> out_high,
                                             legate::Point<NDIM> local_low,
                                             legate::Point<NDIM> local_high)
{
  __shared__ legate::Point<NDIM> block_low[LEGATE_THREADS_PER_BLOCK / LEGATE_WARP_SIZE];
  __shared__ legate::Point<NDIM> block_high[LEGATE_THREADS_PER_BLOCK / LEGATE_WARP_SIZE];

  // Reduce across the warp
  const int lane_id = threadIdx.x & (LEGATE_WARP_SIZE - 1);
  const int warp_id = threadIdx.x >> 5;

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

template <std::int32_t STORE_NDIM,
          std::int32_t POINT_NDIM,
          typename RED_LOW,
          typename RED_HIGH,
          typename InAcc>
__device__ void find_bounding_box_kernel(legate::detail::Unravel<STORE_NDIM> unravel,
                                         std::size_t num_iters,
                                         legate::detail::CUDAReductionBuffer<RED_LOW> out_low,
                                         legate::detail::CUDAReductionBuffer<RED_HIGH> out_high,
                                         legate::AccessorRO<InAcc, STORE_NDIM> in,
                                         legate::Point<POINT_NDIM> identity_low,
                                         legate::Point<POINT_NDIM> identity_high)
{
  auto local_low  = identity_low;
  auto local_high = identity_high;

  for (std::size_t iter = 0; iter < num_iters; ++iter) {
    const auto index = (iter * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < unravel.volume()) {
      auto p = unravel(index);

      if constexpr (std::is_same_v<InAcc, legate::Rect<POINT_NDIM>>) {
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

}  // namespace

}  // namespace detail

#define FIND_BBOX_TPL_INST(STORE_NDIM, POINT_NDIM)                                               \
  extern "C" __global__ void __launch_bounds__(LEGATE_THREADS_PER_BLOCK, LEGATE_MIN_CTAS_PER_SM) \
    legate_find_bounding_box_kernel_rect_##STORE_NDIM##_##POINT_NDIM(                            \
      legate::detail::Unravel<STORE_NDIM> unravel,                                               \
      std::size_t num_iters,                                                                     \
      legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMin<POINT_NDIM>> out_low,   \
      legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMax<POINT_NDIM>> out_high,  \
      legate::AccessorRO<legate::Rect<POINT_NDIM>, STORE_NDIM> in,                               \
      legate::Point<POINT_NDIM> identity_low,                                                    \
      legate::Point<POINT_NDIM> identity_high)                                                   \
  {                                                                                              \
    detail::find_bounding_box_kernel(                                                            \
      unravel, num_iters, out_low, out_high, in, identity_low, identity_high);                   \
  }                                                                                              \
                                                                                                 \
  extern "C" __global__ void __launch_bounds__(LEGATE_THREADS_PER_BLOCK, LEGATE_MIN_CTAS_PER_SM) \
    legate_find_bounding_box_kernel_point_##STORE_NDIM##_##POINT_NDIM(                           \
      legate::detail::Unravel<STORE_NDIM> unravel,                                               \
      std::size_t num_iters,                                                                     \
      legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMin<POINT_NDIM>> out_low,   \
      legate::detail::CUDAReductionBuffer<legate::detail::ElementWiseMax<POINT_NDIM>> out_high,  \
      legate::AccessorRO<legate::Point<POINT_NDIM>, STORE_NDIM> in,                              \
      legate::Point<POINT_NDIM> identity_low,                                                    \
      legate::Point<POINT_NDIM> identity_high)                                                   \
  {                                                                                              \
    detail::find_bounding_box_kernel(                                                            \
      unravel, num_iters, out_low, out_high, in, identity_low, identity_high);                   \
  }

LEGION_FOREACH_NN(FIND_BBOX_TPL_INST)

#undef FIND_BBOX_TPL_INST
