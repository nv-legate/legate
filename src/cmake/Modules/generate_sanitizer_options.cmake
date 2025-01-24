#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# cmake-format: off
# This function takes sanitizer options defined in a file, and pre-processes them into an
# #include-able header that can be consumed by C/C++. The motivation for doing this is
# that we want to keep all of the suppressions and options in one place.
#
# Normally, for C++ builds just using `__default_<sanitizer>_options()` or the like works
# flawlessly. But for Python builds, we have to LD_PRELOAD=/path/to/libasan.so in order
# for ASAN interceptor hooks to work properly. For whatever reason, this also breaks the
# default options functions (as in, they are never consulted, and the options never
# applied). My best guess is that internally ASAN does:
#
# // Note, NULL as first argument means "search current shared object".
# auto user_options = dlsym(NULL, "__default_asan_options");
# if (user_options) {
#   parse_options(user_options());
# }
#
# but since LD_PRELOAD guarantees that ASAN is the *first* library to load, it will not
# have loaded our libraries, and hence `dlsym()` will return NULL.
#
# But that's just a guess.
#
# This function applies the following transformations to the suppression/options file:
#
# 1. Drop all comment lines (those starting with #)...
# 2. ...unless that line contains a second "#", in which case, the line is assumed to
#    contain a C preprocessor directive. In that case, the first # is discarded (to
#    "expose" the directive), and the line is added verbatim to the output.
# 3. Convert all other line entries into string literals.
# 4. Append a chosen delimiter to each line.
#
# So given:
#
# # A comment
# foo:Bar=10
# # #if defined(SOME_MACRO)
# baz:bop
# # #endif
#
# (Assuming the delimiter is "?") this produces:
#
# "foo:Bar=10?"
# #if defined(SOME_MACRO)
# "baz:bop?"
# #endif
# cmake-format: on
function(legate_generate_sanitizer_options)
  list(APPEND CMAKE_MESSAGE_CONTEXT "generate_sanitizer_options")

  set(options)
  set(one_value_args SRC DELIM DEST)
  set(multi_value_args)
  cmake_parse_arguments(_LEGATE_OPT "${options}" "${one_value_args}"
                        "${multi_value_args}" ${ARGN})

  foreach(opt SRC DELIM DEST)
    if(NOT _LEGATE_OPT_${opt})
      message(FATAL_ERROR "Must pass ${opt}")
    endif()
  endforeach()

  file(STRINGS "${_LEGATE_OPT_SRC}" supprs)

  set(ret "")
  foreach(line IN LISTS supprs)
    string(STRIP "${line}" line)
    string(FIND "${line}" "#" first_hash_pos)
    if(first_hash_pos GREATER_EQUAL 0)
      # Some kind of comment, now to determine if the comment contains pre-processor
      # directives
      math(EXPR second_hash_pos "${first_hash_pos} + 1")
      string(SUBSTRING "${line}" ${second_hash_pos} -1 line)
      string(FIND "${line}" "#" second_hash_pos)
      if(second_hash_pos EQUAL -1)
        # Only 1 # in the line means it's a comment
        continue()
      endif()
      list(APPEND ret "${line}")
    else()
      list(APPEND ret "\"${line}${_LEGATE_OPT_DELIM}\"")
    endif()
  endforeach()

  list(JOIN ret "\n" ret)

  file(CONFIGURE OUTPUT # cmake-lint: disable=E1126
       "${_LEGATE_OPT_DEST}" CONTENT "${ret}")
  message(VERBOSE "generated: ${_LEGATE_OPT_DEST}")
endfunction()
