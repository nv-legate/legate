#=============================================================================
# Copyright 2023 NVIDIA Corporation
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

cmake_minimum_required(VERSION 3.22.1 FATAL_ERROR)

project(multi_scalar_out VERSION 0.1 LANGUAGES C CXX)

if (NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(legate_core REQUIRED)

add_executable(multi_scalar_out multi_scalar_out.cc)

target_link_libraries(multi_scalar_out PRIVATE legate::core)

install(TARGETS multi_scalar_out DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/cmake-install")
