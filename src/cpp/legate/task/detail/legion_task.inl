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

#pragma once

// Useful for IDEs
#include "legate/task/detail/legion_task.h"

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
