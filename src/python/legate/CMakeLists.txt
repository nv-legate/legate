#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

list(APPEND CMAKE_MESSAGE_CONTEXT "legate")

# in case we ever need to support msvc
if(MSVC)
  message(FATAL_ERROR "Don't know the flag to silence all MSVC warnings")
endif()
# Silence all compiler warnings for cythonized C/C++ code. We could try and be smart about
# this, and only remove/silence the specific flags that cause warnings, but at the end of
# the day it will be an arms race we cannot win.
add_compile_options(-w)

include("${LEGATE_CMAKE_DIR}/Modules/find_or_configure.cmake")

legate_find_or_configure_init()

legate_find_or_configure(PACKAGE fmt)

# More tomfoolery required because of scikit-build-core. scikit-build-core has a "wheel
# dir" with special directories that map to different things at install time:
#
# cmake-format: off
# lib/ -> $prefix/lib/python-3.12/site-packages/lib
# include/ -> $prefix/lib/python-3.12/site-packages/include
# platlib/ -> $prefix/lib/python-3.12/site-packages
# data/ -> $prefix
# bin/ -> $prefix/lib/python-3.12/site-packages/bin
# cmake-format: on
#
# This is all fine and dandy, but the problem is that our C++ cmake files just want to
# install to $prefix/lib. So the only way to do that is to set scikit-build-core's install
# prefix to point to the "data/" directory. But hold on, our Cython .so's here want to
# live under site-packages, so we need to move them back into the "platlib" directory.
# Hence we need to do this to back out from data/ and into platlib/.
#
# Why on earth scikit-build-core is being foisted on us, when it is so clearly worse than
# its predecessor I do not know...
cmake_path(RELATIVE_PATH SKBUILD_PLATLIB_DIR BASE_DIRECTORY "${SKBUILD_DATA_DIR}"
           OUTPUT_VARIABLE skbuild_platlib_relpath)
message(STATUS "Using scikit-build-core data-to-platlib directory redirect: ${skbuild_platlib_relpath}"
)

define_property(GLOBAL PROPERTY LEGATE_CYTHON_TARGETS BRIEF_DOCS "Legate cython targets")

if(legate_USE_CPROFILE)
  set(CYTHON_FLAGS "${CYTHON_FLAGS} --directive profile=True")
endif()

function(legate_create_cython_modules)
  list(APPEND CMAKE_MESSAGE_CONTEXT "create_cython_modules")

  set(options)
  set(one_value_args)
  set(multi_value_keywords SOURCES)
  cmake_parse_arguments(_LEGATE_CYTHON "${options}" "${one_value_args}"
                        "${multi_value_keywords}" ${ARGN})

  if(NOT _LEGATE_CYTHON_SOURCES)
    message(FATAL_ERROR "Must pass SOURCES source1.pyx [source2.pyx ... sourceN.pyx]")
  endif()

  # rapids_cython_create_modules() docs state that:
  #
  # INSTALL_DIR: The path relative to the installation prefix so that it can be converted
  # to an absolute path in a relocatable way. If not provided, defaults to the path to
  # CMAKE_CURRENT_SOURCE_DIR relative to PROJECT_SOURCE_DIR.
  #
  # Since we live under src/python/legate/foo/bar, but want to eventually install under
  # legate/foo/bar, we need to manually set the install directory. rel_path in this case
  # will be legate/foo/bar.
  cmake_path(RELATIVE_PATH CMAKE_CURRENT_LIST_DIR BASE_DIRECTORY
             "${LEGATE_PYTHON_PKG_ROOT}" OUTPUT_VARIABLE rel_path)
  set(_cython_install_dir "${skbuild_platlib_relpath}/${rel_path}")
  # Account for the different layout of the Python PIP wheels.
  if(LEGATE_BUILD_PIP_WHEELS)
    set(_cython_install_dir "${rel_path}")
  endif()

  rapids_cython_create_modules(CXX ASSOCIATED_TARGETS legate::legate
                               SOURCE_FILES "${_LEGATE_CYTHON_SOURCES}"
                               LINKED_LIBRARIES legate::legate
                                                Python3::Module
                                                # Some Cython modules require delving into
                                                # the private headers of legate, which may
                                                # contain fmt headers. As a result, we
                                                # also need to add the fmt include paths
                                                # etc. This is not done automatically for
                                                # us, since fmt is a PRIVATE dependency of
                                                # legate (as it is not found in any public
                                                # includes).
                                                fmt::fmt-header-only
                               INSTALL_DIR "${_cython_install_dir}")

  foreach(target IN LISTS RAPIDS_CYTHON_CREATED_TARGETS)
    set_target_properties(${target} PROPERTIES LEGATE_INTERNAL_TARGET TRUE)
    target_compile_features(${target} PRIVATE cxx_std_${CMAKE_CXX_STANDARD})
    # Mark python headers as system (in case user has added warning flags, we don't want
    # the python headers to trigger them)
    target_include_directories(${target} SYSTEM PRIVATE ${Python3_INCLUDE_DIRS})
  endforeach()
  set_property(GLOBAL APPEND PROPERTY LEGATE_CYTHON_TARGETS
                                      ${RAPIDS_CYTHON_CREATED_TARGETS})
endfunction()

add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(timing)
