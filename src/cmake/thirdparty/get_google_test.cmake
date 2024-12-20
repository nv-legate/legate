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

include_guard(GLOBAL)

function(find_or_configure_google_test)
  list(APPEND CMAKE_MESSAGE_CONTEXT "google_test")

  include(${rapids-cmake-dir}/cpm/gtest.cmake)

  rapids_cpm_gtest(CPM_ARGS SYSTEM TRUE)
  legate_install_dependencies(TARGETS GTest::gtest GTest::gmock)
  cpm_export_variables(GTest)
endfunction()
