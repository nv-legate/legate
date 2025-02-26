/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/task/task.h>
#include <legate/task/task_context.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/typedefs.h>

namespace legate::io::hdf5::detail {

/**
 * @brief Read HDF5 file into Legate store
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *     - dataset_name: std::string
 *   - inputs:
 *     - buffer: store (any dtype)
 *
 * NB: the store must be contiguous. To make Legate enforce this,
 *     set `policy.exact = true` in `Mapper::store_mappings()`.
 *
 */
class HDF5Read : public LegateTask<HDF5Read> {
 public:
  static constexpr auto TASK_ID = LocalTaskID{legate::detail::CoreTask::IO_HDF5_FILE_READ};

  static constexpr auto CPU_VARIANT_OPTIONS = VariantOptions{}.with_has_side_effect(true);
  static constexpr auto GPU_VARIANT_OPTIONS = VariantOptions{}
                                                .with_elide_device_ctx_sync(true)
                                                .with_has_allocations(true)
                                                .with_has_side_effect(true);
  static constexpr auto OMP_VARIANT_OPTIONS = CPU_VARIANT_OPTIONS;

  static inline const auto TASK_SIGNATURE =  // NOLINT(cert-err58-cpp)
    legate::TaskSignature{}.inputs(0).outputs(1).scalars(2).redops(0).constraints({{}});

  static void cpu_variant(legate::TaskContext context);
  static void omp_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

}  // namespace legate::io::hdf5::detail
