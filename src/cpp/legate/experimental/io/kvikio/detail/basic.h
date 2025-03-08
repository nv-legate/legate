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
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    TaskConfig{LocalTaskID{legate::detail::CoreTask::IO_KVIKIO_FILE_READ}}
      .with_signature(legate::TaskSignature{}.inputs(0).outputs(1).scalars(1).redops(0).constraints(
        {Span<const legate::ProxyConstraint>{}})  // some compilers complain with {{}}
                      )
      .with_variant_options(legate::VariantOptions{}.with_has_side_effect(true));

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
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    TaskConfig{LocalTaskID{legate::detail::CoreTask::IO_KVIKIO_FILE_WRITE}}
      .with_signature(legate::TaskSignature{}.inputs(1).outputs(0).scalars(1).redops(0).constraints(
        {Span<const legate::ProxyConstraint>{}})  // some compilers complain with {{}}
                      )
      .with_variant_options(legate::VariantOptions{}.with_has_side_effect(true));

  static void cpu_variant(legate::TaskContext context);
  static void omp_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

}  // namespace legate::experimental::io::kvikio::detail
