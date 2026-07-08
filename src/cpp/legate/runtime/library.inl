/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate_defines.h>

#include <legate/runtime/library.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

namespace legate {

inline Library::Library(detail::Library* impl) : impl_{impl} {}

template <typename REDOP>
GlobalRedopID Library::register_reduction_operator(LocalRedopID redop_id)
{
  auto legion_redop_id = get_reduction_op_id(redop_id);
#ifndef __CUDACC__
  if (LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    detail::log_legate().warning() << "For the runtime's DMA engine to GPU accelerate reductions, "
                                      "this reduction operator should be registered in a .cu file.";
  }
#endif
  return legion_redop_id;
}

inline bool Library::operator==(const Library& other) const { return impl() == other.impl(); }

inline bool Library::operator!=(const Library& other) const { return !(*this == other); }

inline const detail::Library* Library::impl() const { return impl_; }

inline detail::Library* Library::impl() { return impl_; }

}  // namespace legate
