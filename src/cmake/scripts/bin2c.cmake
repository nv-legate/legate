#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

# This script can be called as a custom cmake command to generate a C file based on a
# generated file in a platform agnostic way such that the platform doesn't need bin2c to
# be available.
foreach(
  var
  VAR_NAME
  IN_FILE
  OUT_CC_FILE
  OUT_H_FILE
  LEGATE_CMAKE_DIR
)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "Must define ${var}")
  endif()
endforeach()

# Read the input file as raw hex values
file(READ "${IN_FILE}" FILE_CONTENTS_HEX HEX)
# Prefix the hex values with 0x
string(
  REGEX REPLACE "([0-9a-f][0-9a-f])"
  "0x\\1,"
  FILE_CONTENTS_HEX
  "${FILE_CONTENTS_HEX}"
)
configure_file("${LEGATE_CMAKE_DIR}/templates/bin2c.cc.in" "${OUT_CC_FILE}" @ONLY)
configure_file("${LEGATE_CMAKE_DIR}/templates/bin2c.h.in" "${OUT_H_FILE}" @ONLY)
