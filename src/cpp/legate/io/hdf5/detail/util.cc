/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/util.h>

#include <legate_defines.h>

#include <legate/utilities/abort.h>
#include <legate/utilities/macros.h>

#include <type_traits>

#if LEGATE_DEFINED(LEGATE_USE_HDF5_VFD_GDS)
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

namespace legate::io::hdf5::detail {

namespace {

class EnableGDS {
 public:
  void apply(::hid_t hid) const noexcept;
};

void EnableGDS::apply(::hid_t hid) const noexcept
{
  if (const auto ret = H5Pset_fapl_gds(hid, 4096, 4096, 16 * 1024 * 1024); ret < 0) {
    LEGATE_ABORT("Error in setting vfd-gds driver to be the I/O filter driver. Error code: ", ret);
  }
}

}  // namespace

HighFive::File open_hdf5_file(const wrapper::HDF5MaybeLockGuard&,
                              const std::string& filepath,
                              bool gds_on,
                              HighFive::File::AccessMode open_mode)
{
  static_assert(!std::is_constructible_v<HighFive::File,
                                         std::string_view,
                                         std::decay_t<decltype(open_mode)>,
                                         HighFive::FileAccessProps>,
                "Can use std::string_view for filepath");
  if (gds_on) {
    auto access_props = HighFive::FileAccessProps::Empty();

    access_props.add(EnableGDS{});
    return HighFive::File{filepath, open_mode, access_props};
  }
  return HighFive::File{filepath, open_mode};
}

}  // namespace legate::io::hdf5::detail
