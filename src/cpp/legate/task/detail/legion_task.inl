/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Useful for IDEs
#include <legate/task/detail/legion_task.h>

namespace legate::detail {

template <typename T>
template <typename U, LegionVariantImpl<U> variant_fn, VariantCode /*variant_kind*/>
/*static*/ void LegionTask<T>::task_wrapper_(const void* args,
                                             std::size_t arglen,
                                             const void* userdata,
                                             std::size_t userlen,
                                             Legion::Processor p)
{
  if constexpr (std::is_same_v<U, void>) {
    Legion::LegionTaskWrapper::legion_task_wrapper<variant_fn>(args, arglen, userdata, userlen, p);
  } else {
    Legion::LegionTaskWrapper::legion_task_wrapper<U, variant_fn>(
      args, arglen, userdata, userlen, p);
  }
}

}  // namespace legate::detail
