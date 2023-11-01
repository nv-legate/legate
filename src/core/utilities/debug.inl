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

#pragma once

#include "core/utilities/debug.h"

#if LegateDefined(LEGATE_USE_CUDA)
#include <cuda_runtime_api.h>
#endif

#include <sstream>

namespace legate {

template <typename T, int DIM>
[[nodiscard]] std::string print_dense_array(const T* base,
                                            const Point<DIM>& extents,
                                            size_t strides[DIM])
{
  T* buf                        = nullptr;
  const auto is_device_only_ptr = [](const void* ptr) {
#if LegateDefined(LEGATE_USE_CUDA)
    cudaPointerAttributes attrs;
    auto res = cudaPointerGetAttributes(&attrs, ptr);
    return res == cudaSuccess && attrs.type == cudaMemoryTypeDevice;
#else
    static_cast<void>(ptr);
    return false;
#endif
  };

  if (is_device_only_ptr(base)) {
    const auto max_different_types = [](const auto& lhs, const auto& rhs) {
      return lhs < rhs ? rhs : lhs;
    };
    size_t num_elems = 0;
    for (size_t dim = 0; dim < DIM; ++dim) {
      num_elems = max_different_types(num_elems, strides[dim] * extents[dim]);
    }
    buf = new T[num_elems];
#if LegateDefined(LEGATE_USE_CUDA)
    auto res = cudaMemcpy(buf, base, num_elems * sizeof(T), cudaMemcpyDeviceToHost);
    assert(res == cudaSuccess);
#endif
    base = buf;
  }
  std::stringstream ss;

  for (int dim = 0; dim < DIM; ++dim) {
    if (strides[dim] != 0) ss << "[";
  }
  ss << *base;

  coord_t offset   = 0;
  Point<DIM> point = Point<DIM>::ZEROES();
  int dim;
  do {
    for (dim = DIM - 1; dim >= 0; --dim) {
      if (strides[dim] == 0) continue;
      if (point[dim] + 1 < extents[dim]) {
        ++point[dim];
        offset += strides[dim];
        ss << ", ";

        for (auto i = dim + 1; i < DIM; ++i) {
          if (strides[i] != 0) ss << "[";
        }
        ss << base[offset];
        break;
      }
      offset -= point[dim] * strides[dim];
      point[dim] = 0;
      ss << "]";
    }
  } while (dim >= 0);
  if (LegateDefined(LEGATE_USE_CUDA)) delete[] buf;  // LEGATE_USE_CUDA
  return ss.str();
}

template <int DIM, typename ACC>
[[nodiscard]] std::string print_dense_array(ACC accessor, const Rect<DIM>& rect)
{
  Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
  size_t strides[DIM];
  const typename ACC::value_type* base = accessor.ptr(rect, strides);
  return print_dense_array(base, extents, strides);
}

}  // namespace legate
