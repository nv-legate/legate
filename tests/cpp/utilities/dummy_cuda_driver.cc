/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <cstdio>
#include <cstdlib>
#include <string_view>

// NOLINTBEGIN
using CUresult   = int;
using cuuint64_t = unsigned long long;

namespace {

CUresult invalid_function(...)
{
  std::fprintf(stderr,
               "Invalid function call in dummy legate CUDA driver module. Consider whether you "
               "actually need to test whether the function does what it does, or whether you just "
               "need to test that *something* exists.");
  std::abort();
  return 1;
}

CUresult cu_init(unsigned int) { return 0; }

}  // namespace

extern "C" CUresult cuGetProcAddress(const char* name, void** fn_ptr, int, cuuint64_t)
{
  // The unit test only tests that cuInit() is callable, so it suffices to only be able to
  // handle that. All other functions need to get a non-NULL pointer (since the loading code
  // checks that), but other than that, they don't need to get a proper function.
  if (std::string_view{name} == "cuInit") {
    *fn_ptr = reinterpret_cast<void*>(cu_init);
  } else {
    *fn_ptr = reinterpret_cast<void*>(invalid_function);
  }
  return 0;
}
// NOLINTEND
