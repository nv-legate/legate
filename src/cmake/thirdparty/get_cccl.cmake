#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

# Use CPM to find or clone CCCL
function(find_or_configure_cccl)
  list(APPEND CMAKE_MESSAGE_CONTEXT "cccl")

  include(${rapids-cmake-dir}/cpm/cccl.cmake)

  if(NOT legate_checked_for_Legion_USE_OpenMP)
    message(FATAL_ERROR "Legion_USE_OpenMP must have been defined already by this point, to properly set the thrust host system settings"
    )
  endif()

  # Change THRUST_DEVICE_SYSTEM for `.cpp` files If we include Thrust in "CUDA mode" in
  # .cc files, that ends up pulling the definition of __half from the CUDA toolkit, and
  # Legion defines a custom __half when compiling outside of nvcc (because CUDA's __half
  # doesn't define any __host__ functions), which causes a conflict.
  if(Legion_USE_OpenMP)
    set(CCCL_THRUST_DEVICE_SYSTEM "OMP")
  else()
    set(CCCL_THRUST_DEVICE_SYSTEM "CPP")
  endif()

  rapids_cpm_cccl(BUILD_EXPORT_SET legate-exports INSTALL_EXPORT_SET legate-exports SYSTEM
                                                                                    TRUE)

  # Workaround for https://github.com/NVIDIA/cccl/issues/5002
  if(Legion_USE_OpenMP)
    get_target_property(opts OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)
    string(REPLACE [[-Xcompiler=SHELL:]] [[SHELL:-Xcompiler=]] opts "${opts}")
    # cmake-lint barfs for some reason here, even though cmake-format is the one that
    # formatted the source to begin with...
    #
    # cmake-format: off
    # path/to/get_cccl.cmake:37,65: [C0307] Bad indentation:
    #                                                        )
    #    ^----BodyNode: 1:0->FlowControlNode: 9:0->BodyNode: 9:32->IfBlockNode:
    #    33:2->BodyNode: 33:23->StatementNode: 36:4->TreeNode: 37:65
    # cmake-format: on
    #
    # cmake-lint: disable=C0307
    set_target_properties(OpenMP::OpenMP_CXX PROPERTIES INTERFACE_COMPILE_OPTIONS
                                                        "${opts}")
  endif()

  legate_export_variables(CCCL)
endfunction()
