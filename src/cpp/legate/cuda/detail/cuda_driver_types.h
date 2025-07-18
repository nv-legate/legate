/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

struct CUctx_st;
struct CUstream_st;
struct CUevent_st;
struct CUfunc_st;
struct CUlib_st;
struct CUkern_st;

namespace legate {

// NOLINTBEGIN
#if defined(_WIN64) || defined(__LP64__)
// Don't use std::uint64_t, we want to match the driver headers exactly
using CUdeviceptr = unsigned long long;  // NOLINT(google-runtime-int)
#else
using CUdeviceptr = unsigned int;
#endif
static_assert(sizeof(CUdeviceptr) == sizeof(void*));

using CUresult   = int;
using CUdevice   = int;
using CUcontext  = ::CUctx_st*;
using CUstream   = ::CUstream_st*;
using CUevent    = ::CUevent_st*;
using CUfunction = ::CUfunc_st*;
using CUlibrary  = ::CUlib_st*;
using CUkernel   = ::CUkern_st*;

enum CUlibraryOption : int;
enum CUjit_option : int;

#ifdef CU_STREAM_PER_THREAD
#define LEGATE_CU_STREAM_DEFAULT CU_STREAM_PER_THREAD
#else
#define LEGATE_CU_STREAM_DEFAULT ((CUstream)0x2)
#endif
// NOLINTEND

}  // namespace legate
