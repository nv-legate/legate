#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_hdf5_vfd_gds)
  list(APPEND CMAKE_MESSAGE_CONTEXT "hdf5_vfd_gds")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(hdf5_vfd_gds version git_url git_tag git_shallow unused)

  # Technically this would also be fixed by the target_link_libraries() below, but if we
  # don't set this before, then we will fail to configure.
  get_target_property(cufile_location CUDA::cuFile LOCATION)
  cmake_path(GET cufile_location PARENT_PATH cufile_location)

  include(GNUInstallDirs)

  rapids_cpm_find(hdf5_vfd_gds "${version}"
                  CPM_ARGS
                  GIT_REPOSITORY "${git_url}"
                  GIT_SHALLOW "${git_shallow}" SYSTEM TRUE
                  GIT_TAG "${git_tag}"
                  OPTIONS "BUILD_TESTING OFF"
                          "BUILD_EXAMPLES OFF"
                          "BUILD_DOCUMENTATION OFF"
                          "HDF5_VFD_GDS_CUFILE_DIR ${cufile_location}"
                          "HDF5_VFD_GDS_INSTALL_BIN_DIR ${CMAKE_INSTALL_BINDIR}"
                          "HDF5_VFD_GDS_INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR}"
                          "HDF5_VFD_GDS_INSTALL_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR}"
                          "HDF5_VFD_GDS_INSTALL_DATA_DIR ${CMAKE_INSTALL_DATAROOTDIR}")

  get_target_property(imported hdf5_vfd_gds IMPORTED)
  if(NOT imported)
    # The CMakeLists for HDF5 VFD GDS are horribly outdated and they do not properly link
    # themselves against cuFile and its headers. So if we built hdf5_vfd_gds ourselves
    # (i.e. not imported), we need to fixup their stuff...
    target_link_libraries(hdf5_vfd_gds CUDA::cuFile)
  endif()

endfunction()
