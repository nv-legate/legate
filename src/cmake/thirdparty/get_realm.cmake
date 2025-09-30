#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

# Use CPM to find or clone Realm
function(find_or_configure_realm)
  list(APPEND CMAKE_MESSAGE_CONTEXT "realm")

  include("${rapids-cmake-dir}/cpm/detail/package_details.cmake")
  rapids_cpm_package_details(Realm version git_url git_tag git_shallow exclude_from_all)

  option(REALM_ENABLE_NVTX "Enabled NVTX" OFF)
  option(REALM_ENABLE_PAPI "Use PAPI for thread profiling" OFF)
  option(REALM_ENABLE_PREALM "Build Realm with support for PRealm" OFF)

  if(legate_ENABLE_SANITIZERS)
    # You can enable more than one sanitizer at a time (for example, ASAN+UBSAN), but
    # Realm doesn't know this. So we pick ASAN here as it is the most widely useful of the
    # bunch.
    set_ifndef(REALM_SANITIZER "ASAN")
  else()
    set_ifndef(REALM_SANITIZER "NONE")
  endif()

  rapids_cpm_find(Realm "${version}"
                  BUILD_EXPORT_SET legate-exports
                  INSTALL_EXPORT_SET legate-exports
                  GLOBAL_TARGETS Realm::Realm
                  CPM_ARGS
                  GIT_SHALLOW "${git_shallow}"
                  GIT_REPOSITORY "${git_url}" SYSTEM TRUE
                  GIT_TAG "${git_tag}"
                  EXCLUDE_FROM_ALL ${exclude_from_all}
                  OPTIONS "REALM_ENABLE_INSTALL ON"
                          "REALM_SANITIZER ${REALM_SANITIZER}"
                          "REALM_ENABLE_UCX ${legate_USE_UCX}"
                          # "REALM_INSTALL_UCX_BOOTSTRAPS ${legate_USE_UCX}"
                          "UCX_BOOTSTRAP_ENABLE_MPI ${legate_USE_UCX}"
                          "REALM_ENABLE_GASNETEX ${Legion_USE_GASNet}"
                          "GASNet_ROOT ${GASNet_ROOT}"
                          "GASNET_CONDUIT ${GASNet_CONDUIT}"
                          "REALM_ENABLE_GASNETEX_WRAPPER ${Legion_USE_GASNETEX_WRAPPER}"
                          "REALM_MAX_DIM ${Legion_MAX_DIM}"
                          "REALM_ENABLE_CUDA ${legate_USE_CUDA}"
                          "REALM_CUDA_DYNAMIC_LOAD ${Legion_CUDA_DYNAMIC_LOAD}"
                          "REALM_ENABLE_NVTX ${REALM_ENABLE_NVTX}"
                          "REALM_ENABLE_HIP OFF"
                          "REALM_ENABLE_LLVM OFF"
                          "REALM_ENABLE_HDF5 OFF"
                          "REALM_ENABLE_MPI ${legate_USE_MPI}"
                          "REALM_ENABLE_OPENMP ${Legion_USE_OpenMP}"
                          "REALM_ENABLE_PAPI ${REALM_ENABLE_PAPI}"
                          "REALM_ENABLE_PREALM ${REALM_ENABLE_PREALM}"
                          "REALM_ENABLE_LIBDL ON"
                          "REALM_ENABLE_PYTHON ${Legion_USE_Python}"
                          "REALM_LOG_LEVEL DEBUG"
                          "INSTALL_SUFFIX -legate"
                          "CMAKE_INSTALL_BINDIR ${legate_DEP_INSTALL_BINDIR}"
                          "CMAKE_INSTALL_INCLUDEDIR ${legate_DEP_INSTALL_INCLUDEDIR}"
                          "CMAKE_SUPPRESS_DEVELOPER_WARNINGS ON")

  # Due to Realm being a direct link-time dependency for downstream users, we cannot stuff
  # it in our deps subtree on install, and instead need to put it at top-level lib/. This
  # runs the risk of clobbering an already existing librealm.so. To avoid this, we name
  # our version of Realm "realm-legate.so".
  if(TARGET realm_cuhook)
    set_target_properties(realm_cuhook PROPERTIES OUTPUT_NAME realm-cuhook-legate)
  endif()

  if(TARGET Realm)
    set_target_properties(Realm PROPERTIES OUTPUT_NAME realm-legate)
  endif()

  if(exclude_from_all)
    legate_install_dependencies(TARGETS Realm::Realm)
  endif()

  legate_export_variables(Realm)
endfunction()
