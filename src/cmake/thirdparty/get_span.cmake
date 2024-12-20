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

function(find_or_configure_span)
  list(APPEND CMAKE_MESSAGE_CONTEXT "span")

  if(CMAKE_CXX_STANDARD GREATER_EQUAL 20)
    include(CheckIncludeFileCXX)

    check_include_file_cxx("span" have_std_span)
    if(have_std_span)
      return()
    endif()
  endif()

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(span version git_url git_tag git_shallow exclude_from_all)

  # Span is strictly header-only so our dependency install is a little unorthodox. Instead
  # of using EXCLUDE_FROM_ALL + custom install to ensure we put it in the right place, we
  # must install() it normally, but point the default CMake install locations to our
  # private directories.
  #
  # The reason for this is that install(TARGETS) (which is what
  # legate_install_dependencies() uses) would not install anything unless the package
  # declares PUBLIC_HEADER or PRIVATE_HEADER. Span does neither of these, so we need to do
  # things this way.
  if(exclude_from_all)
    message(FATAL_ERROR "span must NOT have a truthy EXCLUDE_FROM_ALL (have "
                        "${exclude_from_all}). Setting this to true has no effect, and "
                        "will ensure that the installed package is ill-formed!")
  endif()

  rapids_cpm_find(span "${version}"
                  BUILD_EXPORT_SET legate-exports
                  INSTALL_EXPORT_SET legate-exports
                  CPM_ARGS
                  GIT_REPOSITORY "${git_url}"
                  GIT_SHALLOW "${git_shallow}" SYSTEM TRUE
                  GIT_TAG "${git_tag}"
                  OPTIONS "CMAKE_INSTALL_INCLUDEDIR ${legate_DEP_INSTALL_INCLUDEDIR}"
                          "CMAKE_INSTALL_LIBDIR ${legate_DEP_INSTALL_LIBDIR}")

  include("${rapids-cmake-dir}/export/find_package_root.cmake")
  rapids_export_find_package_root(INSTALL span
                                  [=[${CMAKE_CURRENT_LIST_DIR}/../../legate/deps/cmake]=]
                                  EXPORT_SET legate-exports)
  cpm_export_variables(span)
endfunction()
