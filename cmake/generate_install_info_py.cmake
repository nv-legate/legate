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

execute_process(
  COMMAND ${CMAKE_C_COMPILER}
    -E -DLEGATE_USE_PYTHON_CFFI
    -I "${CMAKE_CURRENT_LIST_DIR}/../src/core"
    -P "${CMAKE_CURRENT_LIST_DIR}/../src/core/legate_c.h"
  ECHO_ERROR_VARIABLE
  OUTPUT_VARIABLE header
  COMMAND_ERROR_IS_FATAL ANY
)

set(libpath "")
configure_file(
  "${CMAKE_CURRENT_LIST_DIR}/../legate/install_info.py.in"
  "${CMAKE_CURRENT_LIST_DIR}/../legate/install_info.py"
@ONLY)
