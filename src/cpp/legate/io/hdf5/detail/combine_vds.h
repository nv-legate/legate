/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/task.h>
#include <legate/task/task_config.h>
#include <legate/task/task_context.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/typedefs.h>

namespace legate::io::hdf5::detail {

class HDF5CombineVDS : public LegateTask<HDF5CombineVDS> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    TaskConfig{LocalTaskID{legate::detail::CoreTask::IO_HDF5_FILE_COMBINE_VDS}}
      .with_signature(TaskSignature{}.inputs(0).outputs(0).scalars(4).redops(0).constraints(
        {Span<const legate::ProxyConstraint>{}}) /* some compilers complain with {{}} */)
      .with_variant_options(
        VariantOptions{}.with_has_side_effect(true).with_elide_device_ctx_sync(true));

  static void cpu_variant(legate::TaskContext context);
};

}  // namespace legate::io::hdf5::detail
