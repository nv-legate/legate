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

#include <legate_defines.h>

#include <legate/experimental/io/hdf5/detail/util.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/macros.h>

#include <fmt/format.h>
#include <stdexcept>
#include <type_traits>

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <H5FDgds.h>
#else
// NOLINTBEGIN
#define H5Pset_fapl_gds(fapl_id, alignment, block_size, cbuf_size) \
  [&]() -> int {                                                   \
    static_cast<void>(fapl_id);                                    \
    static_cast<void>(alignment);                                  \
    static_cast<void>(block_size);                                 \
    static_cast<void>(cbuf_size);                                  \
    return 0;                                                      \
  }()
// NOLINTEND
#endif

namespace legate::experimental::io::hdf5::detail {

[[nodiscard]] HighFive::File open_hdf5_file(const HDF5GlobalLock&,
                                            const std::string& filepath,
                                            bool gds_on)
{
  const auto fapl = HighFive::FileAccessProps{};

  if (LEGATE_DEFINED(LEGATE_USE_CUDA) && gds_on) {
    if (const auto ret = H5Pset_fapl_gds(fapl.getId(), 4096, 4096, 16 * 1024 * 1024); ret < 0) {
      LEGATE_ABORT("Error in setting vfd-gds driver to be the I/O filter driver. Error code: ",
                   ret);
    }
  }

  static_assert(!std::is_constructible_v<HighFive::File,
                                         std::string_view,
                                         std::decay_t<decltype(HighFive::File::ReadOnly)>,
                                         HighFive::FileAccessProps>,
                "Can use std::string_view for filepath");
  return HighFive::File{filepath, HighFive::File::ReadOnly, fapl};
}

}  // namespace legate::experimental::io::hdf5::detail
