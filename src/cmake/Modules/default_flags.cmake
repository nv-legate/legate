#=============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#=============================================================================

include_guard(GLOBAL)

include(CMakePushCheckState)

include(${CMAKE_CURRENT_LIST_DIR}/utilities.cmake)

function(legate_set_default_flags_impl)
  list(APPEND CMAKE_MESSAGE_CONTEXT "set_default_flags")

  set(options IS_LINKER)
  set(one_value_args DEST_VAR LANG)
  set(multi_value_args FLAGS)

  cmake_parse_arguments(_FLAGS "${options}" "${one_value_args}" "${multi_value_args}"
                        ${ARGN})

  if(NOT _FLAGS_DEST_VAR)
    message(FATAL_ERROR "Must pass DEST_VAR")
  endif()

  if(NOT _FLAGS_LANG)
    message(FATAL_ERROR "Must pass LANG")
  endif()

  if(NOT _FLAGS_FLAGS)
    message(VERBOSE "No flags to add to ${_FLAGS_DEST_VAR}")
    return() # nothing to do
  endif()

  get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
  if(NOT ("${_FLAGS_LANG}" IN_LIST languages))
    message(VERBOSE "Language '${_FLAGS_LANG}' not enabled to add flags")
    return()
  endif()

  set(dest)
  foreach(flag IN LISTS _FLAGS_FLAGS)
    set(success FALSE)

    cmake_push_check_state()
    # The sanitizers need to also have the flag passed to the linker. This is kind of a
    # hack, but oh well.
    if(flag MATCHES "sanitize=.*")
      list(APPEND CMAKE_REQUIRED_LINK_OPTIONS "${flag}")
    endif()

    if(_FLAGS_IS_LINKER)
      legate_check_linker_flag(${_FLAGS_LANG} "${flag}" success)
    else()
      legate_check_compiler_flag(${_FLAGS_LANG} "${flag}" success)
    endif()

    cmake_pop_check_state()

    if(success)
      list(APPEND dest "${flag}")
    endif()
  endforeach()

  set(${_FLAGS_DEST_VAR} "${dest}" PARENT_SCOPE)
endfunction()

# Too many statements 51/50
#
# cmake-lint: disable=R0915
function(legate_configure_default_compiler_flags)
  set(default_warning_flags
      "-Wall"
      "-Wextra"
      "-Werror"
      "-Walloca"
      "-Wdeprecated"
      "-Wimplicit-fallthrough"
      "-fdiagnostics-show-template-tree"
      "-Wignored-qualifiers"
      "-Wmissing-field-initializers"
      "-pedantic"
      "-Wsign-compare"
      "-Wshadow"
      "-Wshadow-all"
      "-Warray-bounds-pointer-arithmetic"
      "-Wassign-enum"
      "-Wformat-pedantic"
      "-Wswitch-enum"
      "-Walloc-size"
      "-Walloc-zero"
      "-Wundef"
      "-Wtsan"
      "-Wenum-conversion"
      "-Wpacked")

  if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
    list(APPEND
         default_warning_flags
         # NVC++ warns about not overriding all virtual functions in a derived class. But
         # that, my dear compiler, is the point of making these functions virtual, not
         # pure virtual. So we can disable this worthless warning.
         "--diag_suppress partial_override")
  endif()

  set(default_cxx_flags_debug
      ${default_warning_flags}
      "-g"
      "-O0"
      "-fstack-protector"
      # In classic conda fashion, it sets a bunch of environment variables for you but as
      # usual this just ends up creating more headaches. We don't want FORTIFY_SOURCE
      # because GCC and clang error with:
      #
      # cmake-format: off
      # /tmp/conda-croot/legate/_build_env/x86_64-conda-linux-gnu/sysroot/usr/include/features.h:330:4:
      # error: #warning _FORTIFY_SOURCE requires compiling with optimization (-O)
      # [-Werror=cpp]
      # 330 | #  warning _FORTIFY_SOURCE requires compiling with optimization (-O)
      #     |    ^~~~~~~
      # cmake-format: on
      #
      # Thanks conda, such a great help!
      "-U_FORTIFY_SOURCE")
  set(default_cxx_flags_sanitizer "-fsanitize=address" "-fsanitize=undefined"
                                  "-fsanitize=bounds" "-fno-omit-frame-pointer" "-g")
  set(default_cxx_flags_release ${default_warning_flags} "-O3" "-fstack-protector-strong")
  set(default_cxx_flags_relwithdebinfo ${default_cxx_flags_debug}
                                       ${default_cxx_flags_release})

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(default_cxx_flags ${default_cxx_flags_debug})
  elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(default_cxx_flags ${default_cxx_flags_release})
  elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    set(default_cxx_flags ${default_cxx_flags_relwithdebinfo})
  else()
    set(default_cxx_flags)
  endif()

  if(legate_ENABLE_SANITIZERS)
    list(APPEND default_cxx_flags "${default_cxx_flags_sanitizer}")
  endif()

  if(NOT legate_CXX_FLAGS)
    legate_set_default_flags_impl(LANG CXX DEST_VAR legate_CXX_FLAGS
                                  FLAGS ${default_cxx_flags})
    message(VERBOSE "Set legate_CXX_FLAGS to: ${legate_CXX_FLAGS}")
    set_parent_scope(legate_CXX_FLAGS)
  endif()

  if(legate_ENABLE_SANITIZERS)
    # Legate itself does not use C, but downstream projects might. In this case we want to
    # set the sanitizer flags for them as well, otherwise we get false positives if we
    # work with pointers that they create/destroy.
    foreach(lang CXX C)
      set(flags_tmp)
      # Don't set cache because we still need to de-listify the result first
      legate_set_default_flags_impl(LANG "${lang}" DEST_VAR flags_tmp
                                    FLAGS ${default_cxx_flags_sanitizer})

      list(JOIN flags_tmp " " flags_tmp)
      set(CMAKE_${lang}_FLAGS "${CMAKE_${lang}_FLAGS} ${flags_tmp}")
      message(VERBOSE "Set CMAKE_${lang}_FLAGS to: ${CMAKE_${lang}_FLAGS}")
      set_parent_scope(CMAKE_${lang}_FLAGS)
    endforeach()
  endif()

  if(NOT legate_CUDA_FLAGS)
    set(cuda_flags "${legate_CXX_FLAGS}")
    list(REMOVE_ITEM cuda_flags "-pedantic")
    list(REMOVE_ITEM cuda_flags "-Wpedantic")
    # Remove this, we don't want to wrap it in --compiler-options
    list(REMOVE_ITEM cuda_flags "-U_FORTIFY_SOURCE")

    foreach(flag IN LISTS cuda_flags)
      list(APPEND default_cuda_flags --compiler-options="${flag}")
    endforeach()

    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      list(APPEND default_cuda_flags "-g" "-G")
      # nvcc warning : '--device-debug (-G)' overrides '--generate-line-info (-lineinfo)'
      # ptxas warning : Conflicting options --device-debug and --generate-line-info
      # specified, ignoring --generate-line-info option
      list(REMOVE_ITEM default_cuda_flags "-lineinfo" "--generate-line-info")
      # See C++ flags above for why this is added
      list(APPEND default_cuda_flags -U_FORTIFY_SOURCE)
    elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
      list(REMOVE_ITEM default_cuda_flags "-G" "--device-debug")
      list(APPEND default_cuda_flags "-g" "-lineinfo")
    endif()

    legate_set_default_flags_impl(LANG CUDA DEST_VAR legate_CUDA_FLAGS
                                  FLAGS ${default_cuda_flags})
    message(VERBOSE "Set legate_CUDA_FLAGS to: ${legate_CUDA_FLAGS}")
    set_parent_scope(legate_CUDA_FLAGS)
  endif()
endfunction()

function(legate_configure_default_linker_flags)
  # There are no default linker flags currently.
  set(default_linker_flags)
  if(legate_ENABLE_SANITIZERS)
    list(APPEND default_linker_flags "-fsanitize=address" "-fsanitize=undefined"
         "-fsanitize=bounds" "-fno-sanitize-recover=undefined")
  endif()

  if(NOT legate_LINKER_FLAGS)
    legate_set_default_flags_impl(IS_LINKER LANG CXX DEST_VAR legate_LINKER_FLAGS
                                  FLAGS ${default_linker_flags})
    message(VERBOSE "Set legate_LINKER_FLAGS to: ${legate_LINKER_FLAGS}")
    set_parent_scope(legate_LINKER_FLAGS)
  endif()
endfunction()
