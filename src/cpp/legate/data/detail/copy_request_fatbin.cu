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
__device__ void legate_copy_subdomain_(void* target_ptr,
                                       const void* source_ptr,
                                       std::size_t volume,
                                       const DomainPoint& extents,
                                       const DomainPoint& lo,
                                       const DomainPoint& target_strides,
                                       const DomainPoint& source_strides)
{
  auto* target       = static_cast<VAL*>(target_ptr);
  const auto* source = static_cast<const VAL*>(source_ptr);

  auto idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx >= volume) {
    return;
  }

  DomainPoint point;

  point.dim = extents.dim;
  for (std::int32_t dim = extents.dim - 1; dim > 0; --dim) {
    point[dim] = idx % extents[dim];
    idx /= extents[dim];
  }
  point[0] = idx;
  point += lo;

  target[inner_product(target_strides, point)] = source[inner_product(source_strides, point)];
}

}  // namespace

extern "C" __global__ void legate_copy_subdomain_8(void* target_ptr,
                                                   const void* source_ptr,
                                                   std::size_t volume,
                                                   DomainPoint extents,
                                                   DomainPoint lo,
                                                   DomainPoint target_strides,
                                                   DomainPoint source_strides)
{
  legate_copy_subdomain_<std::uint8_t>(
    target_ptr, source_ptr, volume, extents, lo, target_strides, source_strides);
}

extern "C" __global__ void legate_copy_subdomain_16(void* target_ptr,
                                                    const void* source_ptr,
                                                    std::size_t volume,
                                                    DomainPoint extents,
                                                    DomainPoint lo,
                                                    DomainPoint target_strides,
                                                    DomainPoint source_strides)
{
  legate_copy_subdomain_<std::uint16_t>(
    target_ptr, source_ptr, volume, extents, lo, target_strides, source_strides);
}

extern "C" __global__ void legate_copy_subdomain_32(void* target_ptr,
                                                    const void* source_ptr,
                                                    std::size_t volume,
                                                    DomainPoint extents,
                                                    DomainPoint lo,
                                                    DomainPoint target_strides,
                                                    DomainPoint source_strides)
{
  legate_copy_subdomain_<std::uint32_t>(
    target_ptr, source_ptr, volume, extents, lo, target_strides, source_strides);
}

extern "C" __global__ void legate_copy_subdomain_64(void* target_ptr,
                                                    const void* source_ptr,
                                                    std::size_t volume,
                                                    DomainPoint extents,
                                                    DomainPoint lo,
                                                    DomainPoint target_strides,
                                                    DomainPoint source_strides)
{
  legate_copy_subdomain_<std::uint64_t>(
    target_ptr, source_ptr, volume, extents, lo, target_strides, source_strides);
}

}  // namespace legate::detail
