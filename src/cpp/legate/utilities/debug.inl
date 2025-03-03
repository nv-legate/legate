/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/cuda/cuda.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/debug.h>
#include <legate/utilities/macros.h>

#include <sstream>

namespace legate {

template <typename T, int DIM>
[[nodiscard]] std::string print_dense_array(const T* base,
                                            const Point<DIM>& extents,
                                            const std::size_t (&strides)[DIM])
{
  T* buf                            = nullptr;
  constexpr auto is_device_only_ptr = [](const void* ptr) {
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
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
    std::size_t num_elems = 0;
    for (std::size_t dim = 0; dim < DIM; ++dim) {
      num_elems = max_different_types(num_elems, strides[dim] * extents[dim]);
    }
    buf = new T[num_elems];
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
    auto res = cudaMemcpy(buf, base, num_elems * sizeof(T), cudaMemcpyDeviceToHost);
    LEGATE_CHECK(res == cudaSuccess);
#endif
    base = buf;
  }
  std::stringstream ss;

  for (int dim = 0; dim < DIM; ++dim) {
    if (strides[dim] != 0) {
      ss << "[";
    }
  }
  ss << *base;

  coord_t offset   = 0;
  Point<DIM> point = Point<DIM>::ZEROES();
  int dim;
  do {
    for (dim = DIM - 1; dim >= 0; --dim) {
      if (strides[dim] == 0) {
        continue;
      }
      if (point[dim] + 1 < extents[dim]) {
        ++point[dim];
        offset += strides[dim];
        ss << ", ";

        for (auto i = dim + 1; i < DIM; ++i) {
          if (strides[i] != 0) {
            ss << "[";
          }
        }
        ss << base[offset];
        break;
      }
      offset -= point[dim] * strides[dim];
      point[dim] = 0;
      ss << "]";
    }
  } while (dim >= 0);
  if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    delete[] buf;  // LEGATE_USE_CUDA
  }
  return ss.str();
}

template <int DIM, typename ACC>
[[nodiscard]] std::string print_dense_array(const ACC& accessor, const Rect<DIM>& rect)
{
  const Point<DIM> extents = rect.hi - rect.lo + Point<DIM>::ONES();
  std::size_t strides[DIM];
  const typename ACC::value_type* base = accessor.ptr(rect, strides);
  return print_dense_array(base, extents, strides);
}

}  // namespace legate
