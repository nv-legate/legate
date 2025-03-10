/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/io/detail/task.h>

#include <legate_defines.h>

#include <legate/experimental/io/detail/library.h>
#include <legate/experimental/io/kvikio/detail/basic.h>
#include <legate/experimental/io/kvikio/detail/tile.h>
#include <legate/experimental/io/kvikio/detail/tile_by_offsets.h>
#include <legate/io/hdf5/detail/read.h>

namespace legate::experimental::io::detail {

void register_tasks()
{
  auto&& lib = core_io_library();

  // Kvikio
  kvikio::detail::BasicRead::register_variants(lib);
  kvikio::detail::BasicWrite::register_variants(lib);
  kvikio::detail::TileRead::register_variants(lib);
  kvikio::detail::TileWrite::register_variants(lib);
  kvikio::detail::TileByOffsetsRead::register_variants(lib);
  // HDF5
  if constexpr (LEGATE_DEFINED(LEGATE_USE_HDF5)) {
    legate::io::hdf5::detail::HDF5Read::register_variants(lib);
  }
}

}  // namespace legate::experimental::io::detail
