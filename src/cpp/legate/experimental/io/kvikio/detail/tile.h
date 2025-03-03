/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/task.h>
#include <legate/task/task_context.h>
#include <legate/task/task_signature.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/typedefs.h>

namespace legate::experimental::io::kvikio::detail {

/**
 * @brief Read a tiled Legate store from disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *     - tile_start: tuple of std::uint64_t
 *   - outputs:
 *     - buffer: store (any dtype)
 *
 * NB: the store must be contigues. To make Legate in force this,
 *     set `policy.exact = true` in `Mapper::store_mappings()`.
 *
 */
class TileRead : public LegateTask<TileRead> {
 public:
  static constexpr auto TASK_ID = LocalTaskID{legate::detail::CoreTask::IO_KVIKIO_TILE_READ};

  static inline const auto TASK_SIGNATURE =  // NOLINT(cert-err58-cpp)
    legate::TaskSignature{}.inputs(0).outputs(1).scalars(2).redops(0).constraints(
      {Span<const legate::ProxyConstraint>{}});  // some compilers complain with {{}}

  static void cpu_variant(legate::TaskContext context);
  static void omp_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

/**
 * @brief Write a tiled Legate store to disk using KvikIO
 * Task signature:
 *   - scalars:
 *     - path: std::string
 *     - tile_start: tuple of std::uint64_t
 *   - inputs:
 *     - buffer: store (any dtype)
 *
 * NB: the store must be contigues. To make Legate in force this,
 *     set `policy.exact = true` in `Mapper::store_mappings()`.
 *
 */
class TileWrite : public LegateTask<TileWrite> {
 public:
  static constexpr auto TASK_ID = LocalTaskID{legate::detail::CoreTask::IO_KVIKIO_TILE_WRITE};

  static inline const auto TASK_SIGNATURE =  // NOLINT(cert-err58-cpp)
    legate::TaskSignature{}.inputs(1).outputs(0).scalars(2).redops(0).constraints(
      {Span<const legate::ProxyConstraint>{}});  // some compilers complain with {{}}

  static void cpu_variant(legate::TaskContext context);
  static void omp_variant(legate::TaskContext context);
  static void gpu_variant(legate::TaskContext context);
};

}  // namespace legate::experimental::io::kvikio::detail
