#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

function(find_or_configure_google_test)
  list(APPEND CMAKE_MESSAGE_CONTEXT "google_test")

  include(${rapids-cmake-dir}/cpm/gtest.cmake)

  rapids_cpm_gtest(CPM_ARGS SYSTEM TRUE)
  legate_install_dependencies(TARGETS GTest::gtest GTest::gmock)
  legate_export_variables(GTest)
endfunction()
