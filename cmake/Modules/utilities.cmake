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

macro(legate_string_escape_re_chars output_var input_var)
  # Escapes all special regex characters detailed at
  # https://cmake.org/cmake/help/latest/command/string.html#regex-specification
  string(REGEX REPLACE [[(\.|\-|\+|\*|\^|\$|\?|\||\(|\)|\[|\])]] [[\\\1]] ${output_var}
                       "${${input_var}}")
endmacro()

macro(legate_list_escape_re_chars output_var input_var)
  set(${output_var} ${${input_var}})
  list(TRANSFORM ${output_var} REPLACE [[(\.|\-|\+|\*|\^|\$|\?|\||\(|\)|\[|\])]] [[\\\1]])
endmacro()

function(legate_string_ends_with)
  set(options)
  set(one_value_args SRC ENDS_WITH RESULT_VAR)
  set(multi_value_args)

  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _LEGATE_SRC)
    message(FATAL_ERROR "Must pass SRC")
  endif()

  if(NOT _LEGATE_ENDS_WITH)
    message(FATAL_ERROR "Must pass ENDS_WITH")
  endif()

  if(NOT _LEGATE_RESULT_VAR)
    message(FATAL_ERROR "Must pass RESULT_VAR")
  endif()

  string(STRIP "${_LEGATE_SRC}" src)
  string(STRIP "${_LEGATE_ENDS_WITH}" ends_with)

  legate_string_escape_re_chars(ends_with ends_with)

  if("${src}" MATCHES "${ends_with}$")
    set(${_LEGATE_RESULT_VAR} TRUE PARENT_SCOPE)
  else()
    set(${_LEGATE_RESULT_VAR} FALSE PARENT_SCOPE)
  endif()
endfunction()

#[=======================================================================[
legate_install_from_tree
------------------------

.. code-block:: cmake

  legate_install_from_tree( SRC_ROOT <dirname-or-path>
                            COMMON_PATH <path>
                            DEST_ROOT <dirname-or-path>
                            FILES <files...>)

Generate the installation rules for a set of files that share a common source and
installation directory. This helps avoid mistakes in recreating that directory
hierarchy. For example given

::

    share/some/long/path/a.txt
    share/some/long/path/b.txt
    share/some/long/path/c.txt


That needs to be installed at

::

    ${CMAKE_DATAROOT_DIR}/some/long/path/a.txt
    ${CMAKE_DATAROOT_DIR}/some/long/path/b.txt
    ${CMAKE_DATAROOT_DIR}/some/long/path/c.txt


can be done via

::

    legate_install_from_tree(
      SRC_ROOT share
      COMMON_PATH some/long/path
      DEST_ROOT ${CMAKE_DATAROOT_DIR}
      FILES a.txt b.txt c.txt
    )


Notes
^^^^^
The input files must be found by
``${SRC_ROOT}/${COMMON_PATH}/${SOME_FILE_NAME}``. ``DEST_ROOT`` may be absolute, but per
CMake guidance, it is better to have it be relative to install dir.
#]=======================================================================]
function(legate_install_from_tree)
  list(APPEND CMAKE_MESSAGE_CONTEXT "install_from_tree")

  set(options)
  set(one_value_args SRC_ROOT DEST_ROOT COMMON_PATH)
  set(multi_value_args FILES)

  cmake_parse_arguments(_LEGATE "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _LEGATE_SRC_ROOT)
    message(FATAL_ERROR "Must pass SRC_ROOT")
  endif()

  if(NOT _LEGATE_DEST_ROOT)
    message(FATAL_ERROR "Must pass DEST_ROOT")
  endif()

  if(NOT _LEGATE_COMMON_PATH)
    message(FATAL_ERROR "Must pass COMMON_PATH")
  endif()

  if(NOT _LEGATE_FILES)
    message(FATAL_ERROR "Must pass FILES")
  endif()

  cmake_path(SET src_path NORMALIZE "${_LEGATE_SRC_ROOT}/${_LEGATE_COMMON_PATH}")
  cmake_path(SET dest_path NORMALIZE "${_LEGATE_DEST_ROOT}/${_LEGATE_COMMON_PATH}")

  list(TRANSFORM _LEGATE_FILES PREPEND "${src_path}/")
  install(FILES ${_LEGATE_FILES} DESTINATION "${dest_path}")
endfunction()
