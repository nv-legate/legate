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

#include <stddef.h>

// NOLINTBEGIN
#if defined(_WIN64) || defined(__LP64__)
using CUdeviceptr = unsigned long long;
#else
using CUdeviceptr = unsigned int;
#endif
using CUresult   = int;
using CUdevice   = int;
using CUcontext  = struct CUctx_st*;
using CUstream   = struct CUstream_st*;
using CUfunction = struct CUfunc_st*;
using CUevent    = struct CUevent_st*;
using CUkernel   = struct CUkern_st*;
using CUlibrary  = struct CUlib_st*;

enum CUlibraryOption : int;
enum CUjit_option : int;

extern "C" CUresult cuInit(unsigned int) { return 0; }
extern "C" CUresult cuStreamCreate(CUstream*, unsigned int) { return 0; }
extern "C" CUresult cuGetErrorString(CUresult, const char**) { return 0; }
extern "C" CUresult cuGetErrorName(CUresult, const char**) { return 0; }
extern "C" CUresult cuPointerGetAttributes(void*, int, CUdeviceptr) { return 0; }
extern "C" CUresult cuMemcpyAsync(CUdeviceptr, CUdeviceptr, size_t, CUstream) { return 0; }
extern "C" CUresult cuMemcpy(CUdeviceptr, CUdeviceptr, size_t) { return 0; }
extern "C" CUresult cuStreamDestroy(CUstream) { return 0; }
extern "C" CUresult cuStreamSynchronize(CUstream) { return 0; }
extern "C" CUresult cuCtxSynchronize() { return 0; }
extern "C" CUresult cuEventCreate(CUevent*, unsigned int) { return 0; }
extern "C" CUresult cuEventRecord(CUevent, CUstream) { return 0; }
extern "C" CUresult cuEventSynchronize(CUevent) { return 0; }
extern "C" CUresult cuEventElapsedTime(float*, CUevent, CUevent) { return 0; }
extern "C" CUresult cuEventDestroy(CUevent) { return 0; }
extern "C" CUresult cuCtxGetDevice(CUdevice*) { return 0; }
extern "C" CUresult cuLaunchKernel(
  CUfunction, size_t, size_t, size_t, size_t, size_t, size_t, size_t, CUstream, void**, void**)
{
  return 0;
}
extern "C" CUresult cuLibraryLoadData(
  CUlibrary*, const void*, CUjit_option*, void**, size_t, CUlibraryOption*, void**, size_t)
{
  return 0;
}
extern "C" CUresult cuLibraryGetKernel(CUkernel*, CUlibrary, const char*) { return 0; }
extern "C" CUresult cuLibraryUnload(CUlibrary) { return 0; }

// NOLINTEND
