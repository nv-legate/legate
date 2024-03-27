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

function(legate_default_cpp_install target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "default_cpp_install")

  set(options)
  set(one_value_args EXPORT)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT LEGATE_OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_install")
  endif()

  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Modules/include_rapids.cmake")

  legate_include_rapids()

  rapids_cmake_install_lib_dir(lib_dir)

  install(TARGETS ${target}
          DESTINATION ${lib_dir}
	  EXPORT ${LEGATE_OPT_EXPORT})

  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Modules/debug_symbols.cmake")

  legate_core_debug_syms(${target} INSTALL_DIR ${lib_dir})

  set(final_code_block
    "set(${target}_BUILD_LIBDIR ${CMAKE_BINARY_DIR}/legate_${target})"
  )

  rapids_export(
    INSTALL ${target}
    EXPORT_SET ${LEGATE_OPT_EXPORT}
    GLOBAL_TARGETS ${target}
    NAMESPACE legate::
    LANGUAGES ${ENABLED_LANGUAGES}
  )

  # build export targets
  rapids_export(
    BUILD ${target}
    EXPORT_SET ${LEGATE_OPT_EXPORT}
    GLOBAL_TARGETS ${target}
    NAMESPACE legate::
    FINAL_CODE_BLOCK final_code_block
    LANGUAGES ${ENABLED_LANGUAGES}
  )
endfunction()

function(legate_add_cffi header)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_cffi")

  if (NOT DEFINED CMAKE_C_COMPILER)
    message(FATAL_ERROR "Must enable C language to build Legate projects")
  endif()

  set(options)
  set(one_value_args TARGET PY_PATH)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  # determine full Python path
  if (NOT DEFINED LEGATE_OPT_PY_PATH)
      set(py_path "${CMAKE_CURRENT_SOURCE_DIR}/${LEGATE_OPT_TARGET}")
  elseif(IS_ABSOLUTE LEGATE_OPT_PY_PATH)
    set(py_path "${LEGATE_OPT_PY_PATH}")
  else()
      set(py_path "${CMAKE_CURRENT_SOURCE_DIR}/${LEGATE_OPT_PY_PATH}")
  endif()

  # abbreviate for the function below
  set(target ${LEGATE_OPT_TARGET})
  set(install_info_in
[=[
from pathlib import Path

def get_libpath():
    import os, sys, platform
    join = os.path.join
    exists = os.path.exists
    dirname = os.path.dirname
    cn_path = dirname(dirname(__file__))
    so_ext = {
        "": "",
        "Java": ".jar",
        "Linux": ".so",
        "Darwin": ".dylib",
        "Windows": ".dll"
    }[platform.system()]

    def find_lib(libdir):
        target = f"lib@target@{so_ext}*"
        search_path = Path(libdir)
        matches = [m for m in search_path.rglob(target)]
        if matches:
          return matches[0].parent
        return None

    return (
        find_lib("@libdir@") or
        find_lib(join(dirname(dirname(dirname(cn_path))), "lib")) or
        find_lib(join(dirname(dirname(sys.executable)), "lib")) or
        ""
    )

libpath: str = get_libpath()

header: str = """
  @header@
  void @target@_perform_registration();
"""
]=])
  set(install_info_py_in ${CMAKE_BINARY_DIR}/legate_${target}/install_info.py.in)
  set(install_info_py ${py_path}/install_info.py)
  file(WRITE ${install_info_py_in} "${install_info_in}")

  set(generate_script_content
  [=[
    execute_process(
      COMMAND ${CMAKE_C_COMPILER}
        -E
        -P @header@
      ECHO_ERROR_VARIABLE
      OUTPUT_VARIABLE header
      COMMAND_ERROR_IS_FATAL ANY
    )
    configure_file(
        @install_info_py_in@
        @install_info_py@
        @ONLY)
  ]=])

  set(generate_script ${CMAKE_CURRENT_BINARY_DIR}/gen_install_info.cmake)
  file(CONFIGURE
       OUTPUT ${generate_script}
       CONTENT "${generate_script_content}"
       @ONLY
  )

  if (DEFINED ${target}_BUILD_LIBDIR)
    # this must have been imported from an existing editable build
    set(libdir ${${target}_BUILD_LIBDIR})
  else()
    # libraries are built in a common spot
    set(libdir ${CMAKE_BINARY_DIR}/legate_${target})
  endif()
  add_custom_target("${target}_generate_install_info_py" ALL
    COMMAND ${CMAKE_COMMAND}
      -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
      -Dtarget=${target}
      -Dlibdir=${libdir}
      -P ${generate_script}
    OUTPUT ${install_info_py}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating install_info.py"
    DEPENDS ${header}
  )
endfunction()

function(legate_default_python_install target)
  list(APPEND CMAKE_MESSAGE_CONTEXT "default_python_install")

  set(options)
  set(one_value_args EXPORT)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT LEGATE_OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_python_install")
  endif()

  if (SKBUILD)
    add_library(${target}_python INTERFACE)
    add_library(legate::${target}_python ALIAS ${target}_python)
    target_link_libraries(${target}_python INTERFACE legate::core legate::${target})

    install(TARGETS ${target}_python
            DESTINATION ${lib_dir}
            EXPORT ${LEGATE_OPT_EXPORT})

    include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Modules/include_rapids.cmake")

    legate_include_rapids()
    rapids_export(
      INSTALL ${target}_python
      EXPORT_SET ${LEGATE_OPT_EXPORT}
      GLOBAL_TARGETS ${target}_python
      NAMESPACE legate::
    )
  endif()
endfunction()

function(legate_add_cpp_subdirectory dir)
  list(APPEND CMAKE_MESSAGE_CONTEXT "add_cpp_subdirectory")

  set(options)
  set(one_value_args EXPORT TARGET VERSION)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (NOT LEGATE_OPT_EXPORT)
    message(FATAL_ERROR "Need EXPORT name for legate_default_install")
  endif()

  if (NOT LEGATE_OPT_TARGET)
    message(FATAL_ERROR "Need TARGET name for Legate package")
  endif()

  set(legate_core_version)
  if (LEGATE_OPT_VERSION)
    set(legate_core_version ${LEGATE_OPT_VERSION})
  else()
    if ((PROJECT_NAME STREQUAL "legate_core") OR (PROJECT_NAME STREQUAL "legate_core_python"))
      # called directly by our own cmake lists
      set(legate_core_version "${PROJECT_VERSION}")
    elseif ((CMAKE_PROJECT_NAME STREQUAL "legate_core") OR (CMAKE_PROJECT_NAME STREQUAL "legate_core_python"))
      # our cmake lists are top-level
      set(legate_core_version "${CMAKE_PROJECT_VERSION}")
    endif()
  endif()

  # abbreviate for the function
  set(target ${LEGATE_OPT_TARGET})

  include("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Modules/include_rapids.cmake")

  legate_include_rapids()

  rapids_find_package(legate_core CONFIG
          GLOBAL_TARGETS legate::core
          BUILD_EXPORT_SET ${LEGATE_OPT_EXPORT}
          INSTALL_EXPORT_SET ${LEGATE_OPT_EXPORT})

  if (SKBUILD)
    if (NOT DEFINED ${target}_ROOT)
      set(${target}_ROOT ${CMAKE_SOURCE_DIR}/build)
    endif()
    rapids_find_package(${target} CONFIG
      GLOBAL_TARGETS legate::${target}
      BUILD_EXPORT_SET ${LEGATE_OPT_EXPORT}
      INSTALL_EXPORT_SET ${LEGATE_OPT_EXPORT})
    if (NOT ${target}_FOUND)
      add_subdirectory(${dir} ${CMAKE_BINARY_DIR}/legate_${target})
      legate_default_cpp_install(${target} EXPORT ${LEGATE_OPT_EXPORT})
    else()
      # Make sure the libdir is visible to other functions
      set(${target}_BUILD_LIBDIR "${${target}_BUILD_LIBDIR}" PARENT_SCOPE)
    endif()
  else()
    add_subdirectory(${dir} ${CMAKE_BINARY_DIR}/legate_${target})
    legate_default_cpp_install(${target} EXPORT ${LEGATE_OPT_EXPORT})
  endif()
endfunction()

function(legate_cpp_library_template target output_sources_variable)
  set(file_template
[=[
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "legate.h"

namespace @target@ {

struct Registry {
  static legate::TaskRegistrar& get_registrar();
};

template <typename T, int ID>
struct Task : public legate::LegateTask<T> {
  using Registrar = Registry;
  static constexpr int TASK_ID = ID;
};

}
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/legate_library.h "${file_content}")

  set(file_template
[=[
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate_library.h"

namespace @target@ {

static const char* const library_name = "@target@";

Legion::Logger log_@target@(library_name);

/*static*/ legate::TaskRegistrar& Registry::get_registrar()
{
  static legate::TaskRegistrar registrar;
  return registrar;
}

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);

  Registry::get_registrar().register_all_tasks(context);
}

}  // namespace @target@

extern "C" {

void @target@_perform_registration(void)
{
  @target@::registration_callback();
}

}
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/legate_library.cc "${file_content}")

  set(${output_sources_variable}
    legate_library.h
    legate_library.cc
    PARENT_SCOPE
  )
endfunction()

function(legate_python_library_template py_path)
  set(options)
  set(one_value_args TARGET PY_IMPORT_PATH)
  set(multi_value_args)
  cmake_parse_arguments(
    LEGATE_OPT
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )

  if (DEFINED LEGATE_OPT_TARGET)
    set(target "${LEGATE_OPT_TARGET}")
  else()
    string(REPLACE "/" "_" target "${py_path}")
  endif()

  if (DEFINED LEGATE_OPT_PY_IMPORT_PATH)
    set(py_import_path "${LEGATE_OPT_PY_IMPORT_PATH}")
  else()
    string(REPLACE "/" "." py_import_path "${py_path}")
  endif()

  set(fn_library "${CMAKE_CURRENT_SOURCE_DIR}/${py_path}/library.py")

  set(file_template
    [=[
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from legate.core import (
    get_legate_runtime,
)
import os
import platform
from typing import Any
from ctypes import CDLL, RTLD_GLOBAL

# TODO: Make sure we only have one ffi instance?
from legion_cffi import ffi

def dlopen_no_autoclose(ffi: Any, lib_path: str) -> Any:
    # Use an already-opened library handle, which cffi will convert to a
    # regular FFI object (using the definitions previously added using
    # ffi.cdef), but will not automatically dlclose() on collection.
    lib = CDLL(lib_path, mode=RTLD_GLOBAL)
    return ffi.dlopen(ffi.cast("void *", lib._handle))

class UserLibrary:
    def __init__(self, name: str) -> None:
        self.name = name
        self.shared_object: Any = None

        shared_lib_path = self.get_shared_library()
        if shared_lib_path is not None:
            header = self.get_c_header()
            if header is not None:
                ffi.cdef(header)
            # Don't use ffi.dlopen(), because that will call dlclose()
            # automatically when the object gets collected, thus removing
            # symbols that may be needed when destroying C++ objects later
            # (e.g. vtable entries, which will be queried for virtual
            # destructors), causing errors at shutdown.
            shared_lib = dlopen_no_autoclose(ffi, shared_lib_path)
            self.initialize(shared_lib)
            callback_name = self.get_registration_callback()
            callback = getattr(shared_lib, callback_name)
            callback()
        else:
            self.initialize(None)


    @property
    def cffi(self) -> Any:
        return self.shared_object

    def get_name(self) -> str:
        return self.name

    def get_shared_library(self) -> str:
        from @py_import_path@.install_info import libpath
        return os.path.join(libpath, f"lib@target@{self.get_library_extension()}")

    def get_c_header(self) -> str:
        from @py_import_path@.install_info import header

        return header

    def get_registration_callback(self) -> str:
        return "@target@_perform_registration"

    def initialize(self, shared_object: Any) -> None:
        self.shared_object = shared_object

    def destroy(self) -> None:
        pass

    @staticmethod
    def get_library_extension() -> str:
        os_name = platform.system()
        if os_name == "Linux":
            return ".so"
        elif os_name == "Darwin":
            return ".dylib"
        raise RuntimeError(f"unknown platform {os_name!r}")

user_lib = UserLibrary("@target@")
user_context = get_legate_runtime().find_library(user_lib.get_name())
]=])
  string(CONFIGURE "${file_template}" file_content @ONLY)
  file(WRITE "${fn_library}" "${file_content}")
endfunction()
