/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/divmod.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

namespace {

template <typename VAL, int NDIM>
__device__ __forceinline__ void copy_to_send_buffer_dim_specialized_(
  VAL* __restrict__ target,
  const VAL* __restrict__ source,
  std::size_t tid,
  const DomainPoint& extents,
  const FastDivmodU64& divmod,
  const DomainPoint& strides) noexcept
{
  std::int64_t src_off;

  if constexpr (NDIM == 1) {
    src_off = strides[0] * static_cast<std::int64_t>(tid);
  } else if constexpr (NDIM == 2) {
    std::uint64_t row;
    std::uint64_t col;

    divmod(row, col, tid);
    src_off = row * strides[0] + col * strides[1];
  } else if constexpr (NDIM == 3) {
    auto rem     = static_cast<std::int64_t>(tid);
    const auto k = rem % extents[2];
    rem /= extents[2];
    const auto j = rem % extents[1];
    rem /= extents[1];
    src_off = rem * strides[0] + j * strides[1] + k * strides[2];
  } else {
    auto rem = static_cast<std::int64_t>(tid);
    src_off  = 0;
    for (std::int32_t d = extents.dim - 1; d > 0; --d) {
      const auto coord = rem % extents[d];
      rem /= extents[d];
      src_off += coord * strides[d];
    }
    src_off += rem * strides[0];
  }

  target[tid] = source[src_off];
}

template <typename VAL, int NDIM>
__device__ __forceinline__ void copy_from_recv_buffer_dim_specialized_(
  VAL* __restrict__ target,
  const VAL* __restrict__ source,
  std::size_t tid,
  const DomainPoint& extents,
  const FastDivmodU64& divmod,
  const DomainPoint& strides) noexcept
{
  std::int64_t tgt_off;

  if constexpr (NDIM == 1) {
    tgt_off = strides[0] * static_cast<std::int64_t>(tid);
  } else if constexpr (NDIM == 2) {
    const auto col = static_cast<std::int64_t>(tid) % extents[1];
    const auto row = static_cast<std::int64_t>(tid) / extents[1];
    tgt_off        = row * strides[0] + col * strides[1];
  } else if constexpr (NDIM == 3) {
    auto rem     = static_cast<std::int64_t>(tid);
    const auto k = rem % extents[2];
    rem /= extents[2];
    const auto j = rem % extents[1];
    rem /= extents[1];
    tgt_off = rem * strides[0] + j * strides[1] + k * strides[2];
  } else {
    auto rem = static_cast<std::int64_t>(tid);
    tgt_off  = 0;
    for (std::int32_t d = extents.dim - 1; d > 0; --d) {
      const auto coord = rem % extents[d];
      rem /= extents[d];
      tgt_off += coord * strides[d];
    }
    tgt_off += rem * strides[0];
  }

  target[tgt_off] = source[tid];
}

template <typename VAL>
__device__ void legate_copy_to_send_buffer_(void* target_ptr,
                                            const void* source_ptr,
                                            std::size_t volume,
                                            const DomainPoint& target_extents,
                                            const FastDivmodU64& divmod,
                                            const DomainPoint& source_strides) noexcept
{
  auto* target       = static_cast<VAL*>(target_ptr);
  const auto* source = static_cast<const VAL*>(source_ptr);
  const auto tid     = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid >= volume) {
    return;
  }

  switch (target_extents.dim) {
    case 1: {
      copy_to_send_buffer_dim_specialized_<VAL, 1>(
        target, source, tid, target_extents, divmod, source_strides);
      break;
    }
    case 2: {
      copy_to_send_buffer_dim_specialized_<VAL, 2>(
        target, source, tid, target_extents, divmod, source_strides);
      break;
    }
    case 3: {
      copy_to_send_buffer_dim_specialized_<VAL, 3>(
        target, source, tid, target_extents, divmod, source_strides);
      break;
    }
    default: {  // legate-lint: no-switch-default
      copy_to_send_buffer_dim_specialized_<VAL, 0>(
        target, source, tid, target_extents, divmod, source_strides);
      break;
    }
  }
}

template <typename VAL>
__device__ void legate_copy_from_recv_buffer_(void* target_ptr,
                                              const void* source_ptr,
                                              std::size_t volume,
                                              const DomainPoint& source_extents,
                                              const FastDivmodU64& divmod,
                                              const DomainPoint& target_strides) noexcept
{
  auto* target       = static_cast<VAL*>(target_ptr);
  const auto* source = static_cast<const VAL*>(source_ptr);
  const auto tid     = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid >= volume) {
    return;
  }

  switch (source_extents.dim) {
    case 1: {
      copy_from_recv_buffer_dim_specialized_<VAL, 1>(
        target, source, tid, source_extents, divmod, target_strides);
      break;
    }
    case 2: {
      copy_from_recv_buffer_dim_specialized_<VAL, 2>(
        target, source, tid, source_extents, divmod, target_strides);
      break;
    }
    case 3: {
      copy_from_recv_buffer_dim_specialized_<VAL, 3>(
        target, source, tid, source_extents, divmod, target_strides);
      break;
    }
    default: {  // legate-lint: no-switch-default
      copy_from_recv_buffer_dim_specialized_<VAL, 0>(
        target, source, tid, source_extents, divmod, target_strides);
      break;
    }
  }
}

template <typename VAL>
__device__ void legate_reorder_elements_(void* target_ptr,
                                         const void* source_ptr,
                                         std::size_t volume,
                                         const DomainPoint& target_extents,
                                         const DomainPoint& target_lo,
                                         const DomainPoint& tile_shape,
                                         const DomainPoint& tile_strides,
                                         const DomainPoint& color_strides,
                                         std::int64_t tile_volume)
{
  auto* target       = static_cast<VAL*>(target_ptr);
  const auto* source = static_cast<const VAL*>(source_ptr);

  const auto tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (tid >= volume) {
    return;
  }

  auto idx = tid;
  DomainPoint point;

  point.dim = target_extents.dim;
  for (std::int32_t dim = target_extents.dim - 1; dim > 0; --dim) {
    point[dim] = idx % target_extents[dim];
    idx /= target_extents[dim];
  }
  point[0] = idx;

  std::int64_t in_tile_offset = 0;
  std::int64_t color_offset   = 0;

  // `point` is relative to the target origin, but the tile-major source layout lives in the
  // storage's global coordinate space, so shift by `target_lo` before computing tile offsets.
  for (std::int32_t dim = 0; dim < target_extents.dim; ++dim) {
    const auto global = point[dim] + target_lo[dim];

    in_tile_offset += (global % tile_shape[dim]) * tile_strides[dim];
    color_offset += (global / tile_shape[dim]) * color_strides[dim];
  }

  target[tid] = source[in_tile_offset + (color_offset * tile_volume)];
}

}  // namespace

extern "C" __global__ void legate_copy_to_send_buffer_8(void* target_ptr,
                                                        const void* source_ptr,
                                                        std::size_t volume,
                                                        DomainPoint target_extents,
                                                        FastDivmodU64 divmod,
                                                        DomainPoint source_strides)
{
  legate_copy_to_send_buffer_<std::uint8_t>(
    target_ptr, source_ptr, volume, target_extents, divmod, source_strides);
}

extern "C" __global__ void legate_copy_to_send_buffer_16(void* target_ptr,
                                                         const void* source_ptr,
                                                         std::size_t volume,
                                                         DomainPoint target_extents,
                                                         FastDivmodU64 divmod,
                                                         DomainPoint source_strides)
{
  legate_copy_to_send_buffer_<std::uint16_t>(
    target_ptr, source_ptr, volume, target_extents, divmod, source_strides);
}

extern "C" __global__ void legate_copy_to_send_buffer_32(void* target_ptr,
                                                         const void* source_ptr,
                                                         std::size_t volume,
                                                         DomainPoint target_extents,
                                                         FastDivmodU64 divmod,
                                                         DomainPoint source_strides)
{
  legate_copy_to_send_buffer_<std::uint32_t>(
    target_ptr, source_ptr, volume, target_extents, divmod, source_strides);
}

extern "C" __global__ void legate_copy_to_send_buffer_64(void* target_ptr,
                                                         const void* source_ptr,
                                                         std::size_t volume,
                                                         DomainPoint target_extents,
                                                         FastDivmodU64 divmod,
                                                         DomainPoint source_strides)
{
  legate_copy_to_send_buffer_<std::uint64_t>(
    target_ptr, source_ptr, volume, target_extents, divmod, source_strides);
}

extern "C" __global__ void legate_copy_from_recv_buffer_8(void* target_ptr,
                                                          const void* source_ptr,
                                                          std::size_t volume,
                                                          DomainPoint source_extents,
                                                          FastDivmodU64 divmod,
                                                          DomainPoint target_strides)
{
  legate_copy_from_recv_buffer_<std::uint8_t>(
    target_ptr, source_ptr, volume, source_extents, divmod, target_strides);
}

extern "C" __global__ void legate_copy_from_recv_buffer_16(void* target_ptr,
                                                           const void* source_ptr,
                                                           std::size_t volume,
                                                           DomainPoint source_extents,
                                                           FastDivmodU64 divmod,
                                                           DomainPoint target_strides)
{
  legate_copy_from_recv_buffer_<std::uint16_t>(
    target_ptr, source_ptr, volume, source_extents, divmod, target_strides);
}

extern "C" __global__ void legate_copy_from_recv_buffer_32(void* target_ptr,
                                                           const void* source_ptr,
                                                           std::size_t volume,
                                                           DomainPoint source_extents,
                                                           FastDivmodU64 divmod,
                                                           DomainPoint target_strides)
{
  legate_copy_from_recv_buffer_<std::uint32_t>(
    target_ptr, source_ptr, volume, source_extents, divmod, target_strides);
}

extern "C" __global__ void legate_copy_from_recv_buffer_64(void* target_ptr,
                                                           const void* source_ptr,
                                                           std::size_t volume,
                                                           DomainPoint source_extents,
                                                           FastDivmodU64 divmod,
                                                           DomainPoint target_strides)
{
  legate_copy_from_recv_buffer_<std::uint64_t>(
    target_ptr, source_ptr, volume, source_extents, divmod, target_strides);
}

extern "C" __global__ void legate_reorder_elements_8(void* target_ptr,
                                                     const void* source_ptr,
                                                     std::size_t volume,
                                                     DomainPoint target_extents,
                                                     DomainPoint target_lo,
                                                     DomainPoint tile_shape,
                                                     DomainPoint tile_strides,
                                                     DomainPoint color_strides,
                                                     std::int64_t tile_volume)
{
  legate_reorder_elements_<std::uint8_t>(target_ptr,
                                         source_ptr,
                                         volume,
                                         target_extents,
                                         target_lo,
                                         tile_shape,
                                         tile_strides,
                                         color_strides,
                                         tile_volume);
}

extern "C" __global__ void legate_reorder_elements_16(void* target_ptr,
                                                      const void* source_ptr,
                                                      std::size_t volume,
                                                      DomainPoint target_extents,
                                                      DomainPoint target_lo,
                                                      DomainPoint tile_shape,
                                                      DomainPoint tile_strides,
                                                      DomainPoint color_strides,
                                                      std::int64_t tile_volume)
{
  legate_reorder_elements_<std::uint16_t>(target_ptr,
                                          source_ptr,
                                          volume,
                                          target_extents,
                                          target_lo,
                                          tile_shape,
                                          tile_strides,
                                          color_strides,
                                          tile_volume);
}

extern "C" __global__ void legate_reorder_elements_32(void* target_ptr,
                                                      const void* source_ptr,
                                                      std::size_t volume,
                                                      DomainPoint target_extents,
                                                      DomainPoint target_lo,
                                                      DomainPoint tile_shape,
                                                      DomainPoint tile_strides,
                                                      DomainPoint color_strides,
                                                      std::int64_t tile_volume)
{
  legate_reorder_elements_<std::uint32_t>(target_ptr,
                                          source_ptr,
                                          volume,
                                          target_extents,
                                          target_lo,
                                          tile_shape,
                                          tile_strides,
                                          color_strides,
                                          tile_volume);
}

extern "C" __global__ void legate_reorder_elements_64(void* target_ptr,
                                                      const void* source_ptr,
                                                      std::size_t volume,
                                                      DomainPoint target_extents,
                                                      DomainPoint target_lo,
                                                      DomainPoint tile_shape,
                                                      DomainPoint tile_strides,
                                                      DomainPoint color_strides,
                                                      std::int64_t tile_volume)
{
  legate_reorder_elements_<std::uint64_t>(target_ptr,
                                          source_ptr,
                                          volume,
                                          target_extents,
                                          target_lo,
                                          tile_shape,
                                          tile_strides,
                                          color_strides,
                                          tile_volume);
}

}  // namespace legate::detail
