/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include "core/utilities/abort.h"
#include "core/utilities/assert.h"
#include "core/utilities/cpp_version.h"
#include "core/utilities/defined.h"

#include "legion.h"

#ifdef __CUDACC__
#define LEGATE_HOST_DEVICE __host__ __device__
#else
#define LEGATE_HOST_DEVICE
#endif

#ifndef LEGION_REDOP_HALF
#error "Legate needs Legion to be compiled with -DLEGION_REDOP_HALF"
#endif

#if !LegateDefined(LEGATE_USE_CUDA)
#ifdef LEGION_USE_CUDA
#define LEGATE_USE_CUDA 1
#endif
#endif

#if !LegateDefined(LEGATE_USE_OPENMP)
#ifdef REALM_USE_OPENMP
#define LEGATE_USE_OPENMP 1
#endif
#endif

#if !LegateDefined(LEGATE_USE_NETWORK)
#if defined(REALM_USE_GASNET1) || defined(REALM_USE_GASNETEX) || defined(REALM_USE_MPI) || \
  defined(REALM_USE_UCX)
#define LEGATE_USE_NETWORK 1
#endif
#endif

#ifdef LEGION_BOUNDS_CHECKS
#define LEGATE_BOUNDS_CHECKS 1
#endif

#define LEGATE_MAX_DIM LEGION_MAX_DIM

// backwards compatibility
#if defined(DEBUG_LEGATE) && !LegateDefined(LEGATE_USE_DEBUG)
#define LEGATE_USE_DEBUG 1
#endif

// TODO(wonchanl): 2022-10-04: Work around a Legion bug, by not instantiating futures on
// framebuffer.
#define LEGATE_NO_FUTURES_ON_FB 1
