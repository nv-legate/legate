/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

namespace {

__device__ std::int64_t inner_product(const DomainPoint& rhs1, const DomainPoint& rhs2) noexcept
{
  std::int64_t result = 0;

  for (std::int32_t dim = 0; dim < rhs1.dim; ++dim) {
    result += rhs1[dim] * rhs2[dim];
  }

  return result;
}

template <typename VAL>
__device__ void legate_copy_to_send_buffer_(void* target_ptr,
                                            const void* source_ptr,
                                            std::size_t volume,
                                            const DomainPoint& target_extents,
                                            const DomainPoint& source_strides)
{
  auto* target       = static_cast<VAL*>(target_ptr);
  const auto* source = static_cast<const VAL*>(source_ptr);

  auto idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx >= volume) {
    return;
  }

  DomainPoint point;

  point.dim = target_extents.dim;
  for (std::int32_t dim = target_extents.dim - 1; dim > 0; --dim) {
    point[dim] = idx % target_extents[dim];
    idx /= target_extents[dim];
  }
  point[0] = idx;

  target[idx] = source[inner_product(source_strides, point)];
}

template <typename VAL>
__device__ void legate_copy_from_recv_buffer_(void* target_ptr,
                                              const void* source_ptr,
                                              std::size_t volume,
                                              const DomainPoint& source_extents,
                                              const DomainPoint& target_strides)
{
  auto* target       = static_cast<VAL*>(target_ptr);
  const auto* source = static_cast<const VAL*>(source_ptr);

  auto idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx >= volume) {
    return;
  }

  DomainPoint point;

  point.dim = source_extents.dim;
  for (std::int32_t dim = source_extents.dim - 1; dim > 0; --dim) {
    point[dim] = idx % source_extents[dim];
    idx /= source_extents[dim];
  }
  point[0] = idx;

  target[inner_product(target_strides, point)] = source[idx];
}

}  // namespace

extern "C" __global__ void legate_copy_to_send_buffer_8(void* target_ptr,
                                                        const void* source_ptr,
                                                        std::size_t volume,
                                                        DomainPoint target_extents,
                                                        DomainPoint source_strides)
{
  legate_copy_to_send_buffer_<std::uint8_t>(
    target_ptr, source_ptr, volume, target_extents, source_strides);
}

extern "C" __global__ void legate_copy_to_send_buffer_16(void* target_ptr,
                                                         const void* source_ptr,
                                                         std::size_t volume,
                                                         DomainPoint target_extents,
                                                         DomainPoint source_strides)
{
  legate_copy_to_send_buffer_<std::uint16_t>(
    target_ptr, source_ptr, volume, target_extents, source_strides);
}

extern "C" __global__ void legate_copy_to_send_buffer_32(void* target_ptr,
                                                         const void* source_ptr,
                                                         std::size_t volume,
                                                         DomainPoint target_extents,
                                                         DomainPoint source_strides)
{
  legate_copy_to_send_buffer_<std::uint32_t>(
    target_ptr, source_ptr, volume, target_extents, source_strides);
}

extern "C" __global__ void legate_copy_to_send_buffer_64(void* target_ptr,
                                                         const void* source_ptr,
                                                         std::size_t volume,
                                                         DomainPoint target_extents,
                                                         DomainPoint source_strides)
{
  legate_copy_to_send_buffer_<std::uint64_t>(
    target_ptr, source_ptr, volume, target_extents, source_strides);
}

extern "C" __global__ void legate_copy_from_recv_buffer_8(void* target_ptr,
                                                          const void* source_ptr,
                                                          std::size_t volume,
                                                          DomainPoint source_extents,
                                                          DomainPoint target_strides)
{
  legate_copy_from_recv_buffer_<std::uint8_t>(
    target_ptr, source_ptr, volume, source_extents, target_strides);
}

extern "C" __global__ void legate_copy_from_recv_buffer_16(void* target_ptr,
                                                           const void* source_ptr,
                                                           std::size_t volume,
                                                           DomainPoint source_extents,
                                                           DomainPoint target_strides)
{
  legate_copy_from_recv_buffer_<std::uint16_t>(
    target_ptr, source_ptr, volume, source_extents, target_strides);
}

extern "C" __global__ void legate_copy_from_recv_buffer_32(void* target_ptr,
                                                           const void* source_ptr,
                                                           std::size_t volume,
                                                           DomainPoint source_extents,
                                                           DomainPoint target_strides)
{
  legate_copy_from_recv_buffer_<std::uint32_t>(
    target_ptr, source_ptr, volume, source_extents, target_strides);
}

extern "C" __global__ void legate_copy_from_recv_buffer_64(void* target_ptr,
                                                           const void* source_ptr,
                                                           std::size_t volume,
                                                           DomainPoint source_extents,
                                                           DomainPoint target_strides)
{
  legate_copy_from_recv_buffer_<std::uint64_t>(
    target_ptr, source_ptr, volume, source_extents, target_strides);
}

}  // namespace legate::detail
