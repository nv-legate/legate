#=============================================================================
# Copyright 2022 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=============================================================================

option(FIND_LEGATE_CORE_CPP "Search for existing legate_core C++ installations before defaulting to local files"
       OFF)

# If the user requested it we attempt to find RMM.
if(FIND_LEGATE_CORE_CPP)
  find_package(legate_core ${legate_core_version})
else()
  set(legate_core_FOUND OFF)
endif()

if(NOT legate_core_FOUND)
  set(SKBUILD OFF)
  set(Legion_USE_Python ON)
  set(Legion_BUILD_BINDINGS ON)
  add_subdirectory(. legate_core-cpp)
  set(SKBUILD ON)
endif()

execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -E -DLEGATE_USE_PYTHON_CFFI
    -I "${CMAKE_CURRENT_SOURCE_DIR}/core/src"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/src/core/legate_c.h"
  ECHO_ERROR_VARIABLE
  OUTPUT_VARIABLE header
  COMMAND_ERROR_IS_FATAL ANY
)

set(libpath "${CMAKE_CURRENT_BINARY_DIR}/legate_core-cpp/lib")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/legate/core/install_info/__init__.py.in"
  "${CMAKE_CURRENT_SOURCE_DIR}/legate/core/install_info/__init__.py"
@ONLY)

set(install_code_string "")
string(APPEND install_code_string "set(libpath \"\")\n")
string(APPEND install_code_string "set(header [=[${header}]=])\n")
string(APPEND install_code_string "configure_file(\n")
string(APPEND install_code_string "\"${CMAKE_CURRENT_SOURCE_DIR}/legate/core/install_info/__init__.py.in\"\n")
string(APPEND install_code_string "\"\${CMAKE_INSTALL_PREFIX}/legate/core/install_info/__init__.py\"\n")
string(APPEND install_code_string "@ONLY)\n")

install(CODE "${install_code_string}")
