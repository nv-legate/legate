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

# Use CPM to find or clone thrust
function(find_or_configure_thrust)
    include(${rapids-cmake-dir}/cpm/thrust.cmake)

    # Workaround for:
    #
    # 1. debug-clang and debug-sanitizer-clang presets add -Wdeprecated-builtins to C++
    #    flags.
    # 2. We include thrust headers as part of src/core/utilities/detail/enumerate.h.
    # 3. Without SYSTEM TRUE, those headers are added to our build line via
    #    -I/build/<...>/_deps/thrust/include.
    # 4. Those thrust headers (somewhere) internally use __has_trivial_constructor(T),
    #    which is a compiler builtin to implement std::is_trivially_constructible.
    # 5. __has_trivial_constructor() is deprecated in favor of
    #    __is_trivially_constructible().
    # 6. Boom now we get warnings (which are errors because we compile with -Werror).
    #
    # If we use SYSTEM TRUE, then we include the headers using -isystem, which silences
    # these warnings.
    #
    # We cannot, however, do this everywhere, in particular if we have GPUs enabled. In
    # this case NVCC injects -I/path/to/its/own/thrust into our compile line. If both
    # -I/path/to/some/header and -isystem /other/path/to/some/header exist, then -I
    # overrules the -isystem variant. In our case, this leads to:
    #
    # home/coder/.conda/envs/legate/targets/x86_64-linux/include/thrust/system/cuda/config.h:116:2:
    # error: #error The version of CUB in your include path is not compatible with this
    # release of Thrust. CUB is now included in the CUDA Toolkit, so you no longer need to
    # use your own checkout of CUB. Define THRUST_IGNORE_CUB_VERSION_CHECK to ignore this.
    #
    # There is also the issue of conda doing its usual environment variable shenanigans,
    # so to be safe we only do this for clang, since we don't yet test that in CI :).
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(thrust_cpm_args "SYSTEM" "TRUE")
    else()
      set(thrust_cpm_args)
    endif()

    rapids_cpm_thrust(
      NAMESPACE          legate
      BUILD_EXPORT_SET   legate-core-exports
      INSTALL_EXPORT_SET legate-core-exports
      CPM_ARGS
        ${thrust_cpm_args}
    )
endfunction()

find_or_configure_thrust()
