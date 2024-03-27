#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#=============================================================================

function(find_or_configure_nccl)

    if(TARGET NCCL::NCCL)
        return()
    endif()

    rapids_find_generate_module(NCCL
        HEADER_NAMES  nccl.h
        LIBRARY_NAMES nccl
    )

    # Currently NCCL has no CMake build-system so we require
    # it built and installed on the machine already
    rapids_find_package(NCCL REQUIRED)

endfunction()

find_or_configure_nccl()
