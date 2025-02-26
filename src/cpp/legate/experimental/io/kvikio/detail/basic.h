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

namespace legate::experimental::io::kvikio::detail {

/**
 * @brief Read a Legate store from disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *   - outputs:
 *     - buffer: 1d store (any dtype)
 */
class BasicRead : public LegateTask<BasicRead> {
 public:
  static constexpr auto TASK_ID = LocalTaskID{legate::detail::CoreTask::IO_KVIKIO_FILE_READ};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_side_effect(true);
  static constexpr auto OMP_VARIANT_OPTIONS = CPU_VARIANT_OPTIONS;
  static constexpr auto GPU_VARIANT_OPTIONS = CPU_VARIANT_OPTIONS;

  static inline const auto TASK_SIGNATURE =  // NOLINT(cert-err58-cpp)
    legate::TaskSignature{}.inputs(0).outputs(1).scalars(1).redops(0).constraints({{}});

  static void cpu_variant(legate::TaskContext context);
  static void omp_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

/**
 * @brief Write a Legate store to disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *   - inputs:
 *     - buffer: 1d store (any dtype)
 * NB: the file must exist before running this task because in order to support
 *     access from multiple processes, this task opens the file in "r+" mode.
 */
class BasicWrite : public LegateTask<BasicWrite> {
 public:
  static constexpr auto TASK_ID = LocalTaskID{legate::detail::CoreTask::IO_KVIKIO_FILE_WRITE};

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_side_effect(true);
  static constexpr auto OMP_VARIANT_OPTIONS = CPU_VARIANT_OPTIONS;
  static constexpr auto GPU_VARIANT_OPTIONS = CPU_VARIANT_OPTIONS;

  static inline const auto TASK_SIGNATURE =  // NOLINT(cert-err58-cpp)
    legate::TaskSignature{}.inputs(1).outputs(0).scalars(1).redops(0).constraints({{}});

  static void cpu_variant(legate::TaskContext context);
  static void omp_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

}  // namespace legate::experimental::io::kvikio::detail
