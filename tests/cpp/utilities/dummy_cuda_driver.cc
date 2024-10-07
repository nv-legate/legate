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
using CUresult  = int;
using CUdevice  = int;
using CUcontext = struct CUctx_st*;
using CUstream  = struct CUstream_st*;

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
// NOLINTEND
