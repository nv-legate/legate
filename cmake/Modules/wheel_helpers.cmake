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

function(install_imported_rt_deps)
  set(options)
  set(one_value_args TARGET RT_DEPS RUNTIME_DESTINATION LIBRARY_DESTINATION)
  set(multi_value_args)
  cmake_parse_arguments(LG_CORE "${options}" "${one_value_args}" "${multi_value_args}" ${ARGN})

  if(NOT LG_CORE_TARGET)
    message(FATAL_ERROR "Must supply TARGET")
  endif()
  if(NOT TARGET ${LG_CORE_TARGET})
    message(FATAL_ERROR "LG_CORE_TARGET must be a TARGET")
  endif()

  if(NOT LG_CORE_RT_DEPS)
    message(FATAL_ERROR "Must supply RT_DEPS")
  endif()

  if((NOT LG_CORE_LIBRARY_DESTINATION) AND (NOT LG_CORE_RUNTIME_DESTINATION))
    message(FATAL_ERROR "Must supply either LIBRARY_DESTINATION or RUNTIME_DESTINATION: ${LG_CORE_RUNTIME_DESTINATION}, ${LG_CORE_LIBRARY_DESTINATION}")
  endif()

  if(LG_CORE_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR "Unhandled extra arguments found: '${LG_CORE_UNPARSED_ARGUMENTS}'"
    )
  endif()

  set(destination_args)
  if(LG_CORE_LIBRARY_DESTINATION)
    set(destination_args LIBRARY DESTINATION ${LG_CORE_LIBRARY_DESTINATION})
  endif()
  if(LG_CORE_RUNTIME_DESTINATION)
    set(destination_args RUNTIME DESTINATION ${LG_CORE_RUNTIME_DESTINATION})
  endif()

  # HACK HACK HACK
  # TODO(jfaibussowit):
  # Use the same mechanism as cuDF do
  # (https://github.com/rapidsai/cudf/blob/branch-24.04/python/cudf/CMakeLists.txt#L78-L92)
  # to manage installation of all the dependent libs.
  #
  # We currently can't because the Legion python bindings require installing not just
  # libraries, but also binaries and the cuDF function
  # (install_aliased_imported_targets()) only handles libraries.
  install(
    IMPORTED_RUNTIME_ARTIFACTS ${LG_CORE_TARGET}
    RUNTIME_DEPENDENCY_SET     ${LG_CORE_RT_DEPS}
  )

  macro(escape_re_chars some_list)
    # Escapes all special regex characters detailed at
    # https://cmake.org/cmake/help/latest/command/string.html#regex-specification
    list(TRANSFORM ${some_list}
      REPLACE [[(\.|\-|\+|\*|\^|\$|\?|\||\(|\)|\[|\])]] [[\\\1]]
    )
  endmacro()
  # The following is some absolute buffoonery. We would like to exclude the runtime libs
  # that are actually system libraries or coming from a conda env from the installed
  # libs. Otherwise, those will end up being added to the wheel, which means they will also
  # be _deleted_ when that wheel is uninstalled. Not good.
  set(
    prefix_path_excl
    ${CMAKE_SYSTEM_PREFIX_PATH} # exclude purely system packages
    ${CMAKE_PREFIX_PATH} $ENV{CMAKE_PREFIX_PATH} # either one possibly set by rapids_cmake
  )
  if(WIN32)
    message(
      FATAL_ERROR
      "Figure out path separator for windows, i.e. equivalent of unix ':' for PATH=path_1:path_2"
    )
  endif()
  string(REPLACE ":" ";" prefix_path_excl "${prefix_path_excl}")
  list(REMOVE_DUPLICATES prefix_path_excl)
  list(TRANSFORM prefix_path_excl APPEND "/${CMAKE_INSTALL_LIBDIR}/")
  # Yes, this is the nuclear option. For whatever reason, it is not enough to ignore the
  # directory that these system libraries are in on Linux, we must explicitly enumerate all
  # of the libs we wish to ignore.
  foreach(_dir IN LISTS prefix_path_excl)
    file(GLOB lib_names
      LIST_DIRECTORIES false
      RELATIVE ${_dir}
      "${_dir}*${CMAKE_SHARED_LIBRARY_SUFFIX}*"
    )
    list(APPEND lib_excl "${lib_names}")
  endforeach()
  list(REMOVE_DUPLICATES lib_excl)
  escape_re_chars(lib_excl)
  if(Legion_USE_CUDA)
    # libcuda.so is a bit of a special snowflake.
    #
    # Unlike most of the other libraries that ship with CUDA SDK, libcuda.so is provided by
    # the NVIDIA driver, which is only installed on the machines where NVIDIA GPUs are
    # present. This is often not the case for the machines where one builds CUDA apps.
    #
    # So, in order to be able to build a functional CUDA app which uses the driver API, the
    # executable has to be linked with stubs/libcuda.so. The stub is essentially an
    # interface library, which only provides the symbols and allows the linker to finish
    # linking the executable w/o complaining about the missing
    # symbols. DT_SONAME=libcuda.so.1 of the stub is intentionally does not match the file
    # name libcuda.so, because we do not want dynamic linker to ever load stub/libcuda.so if
    # we were to run the executable linked with it.
    #
    # Instead, when the executable is run, dynamic linker will go searching for libcuda.so.1
    # among the shared libraries in the standard search path. On machines where NVIDIA
    # driver is installed, it will find the real libcuda.so.1.X.Y provided by the driver
    # vX.Y. On machines w/o the GPU the execution will be expected to fail due to the
    # missing libcuda.so.
    #
    # So... the point here is that libcuda.<whatever> should NEVER be considered an
    # installable runtime artifact!
    list(APPEND lib_excl "libcuda\\${CMAKE_SHARED_LIBRARY_SUFFIX}.*")
  endif()
  if(UNIX)
    # Linux's linker, which lives under lib64, and hence won't get picked up in the globs
    # above (which only search under lib, don't let CMAKE_INSTALL_LIBDIR fool you)
    list(APPEND lib_excl "ld-.*\\${CMAKE_SHARED_LIBRARY_SUFFIX}.*")
    # These also appear to live in super special places, and for whatever reason don't get
    # picked up sometimes on unix systems. So we name them explicitly
    list(APPEND lib_excl
      "libc\\${CMAKE_SHARED_LIBRARY_SUFFIX}.*" "libm\\${CMAKE_SHARED_LIBRARY_SUFFIX}.*")
  endif()
  message(STATUS "Excluding the following libs: ${lib_excl}")

  escape_re_chars(prefix_path_excl)
  # This must be last! We must not escape the appended '.*'
  list(TRANSFORM prefix_path_excl APPEND ".*")
  message(STATUS "Excluding the following paths: ${prefix_path_excl}")

  install(
    RUNTIME_DEPENDENCY_SET ${LG_CORE_RT_DEPS}
    PRE_EXCLUDE_REGEXES    ${lib_excl}
    POST_EXCLUDE_REGEXES   ${prefix_path_excl}
    ${destination_args}
  )
endfunction()
