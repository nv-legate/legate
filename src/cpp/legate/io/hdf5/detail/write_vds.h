/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/proxy.h>
#include <legate/task/task.h>
#include <legate/task/task_config.h>
#include <legate/task/task_context.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/typedefs.h>

namespace legate::io::hdf5::detail {

class HDF5WriteVDS : public LegateTask<HDF5WriteVDS> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    TaskConfig{LocalTaskID{legate::detail::CoreTask::IO_HDF5_FILE_WRITE_VDS}}
      .with_signature(TaskSignature{}.inputs(1).outputs(0).scalars(2).redops(1).constraints(
        {Span<const legate::ProxyConstraint>{}}) /* some compilers complain with {{}} */)
      .with_variant_options(
        VariantOptions{}.with_has_side_effect(true).with_elide_device_ctx_sync(true));

  static constexpr auto GPU_VARIANT_OPTIONS = VariantOptions{}
                                                .with_elide_device_ctx_sync(true)
                                                .with_has_allocations(true)
                                                .with_has_side_effect(true);

  static void cpu_variant(legate::TaskContext context);
  static void omp_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

}  // namespace legate::io::hdf5::detail
