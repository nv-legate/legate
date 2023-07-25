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

    rapids_cpm_thrust(NAMESPACE legate
                      BUILD_EXPORT_SET legate-core-exports
                      INSTALL_EXPORT_SET legate-core-exports)
endfunction()

find_or_configure_thrust()
