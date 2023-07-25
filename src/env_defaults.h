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

// These values are copied manually in legate.settings and there is a Python
// unit test that will maintain that these values and the Python settings
// values agree. If these values are modified, the corresponding Python values
// must also be updated.

// 1 << 20 (need actual number for python to parse)
#define MIN_GPU_CHUNK_DEFAULT 1048576
#define MIN_GPU_CHUNK_TEST 2

// 1 << 14 (need actual number for python to parse)
#define MIN_CPU_CHUNK_DEFAULT 16384
#define MIN_CPU_CHUNK_TEST 2

// 1 << 17 (need actual number for python to parse)
#define MIN_OMP_CHUNK_DEFAULT 131072
#define MIN_OMP_CHUNK_TEST 2

#define WINDOW_SIZE_DEFAULT 1
#define WINDOW_SIZE_TEST 1

#ifdef DEBUG_LEGATE
// In debug mode, the default is always block on tasks that can throw exceptions
#define MAX_PENDING_EXCEPTIONS_DEFAULT 1
#else
#define MAX_PENDING_EXCEPTIONS_DEFAULT 64
#endif
#define MAX_PENDING_EXCEPTIONS_TEST 1

#define PRECISE_EXCEPTION_TRACE_DEFAULT 0
#define PRECISE_EXCEPTION_TRACE_TEST 0

#define FIELD_REUSE_FRAC_DEFAULT 256
#define FIELD_REUSE_FRAC_TEST 256

#define FIELD_REUSE_FREQ_DEFAULT 32
#define FIELD_REUSE_FREQ_TEST 32

#define MAX_LRU_LENGTH_DEFAULT 5
#define MAX_LRU_LENGTH_TEST 1

#define DISABLE_MPI_DEFAULT 0
#define DISABLE_MPI_TEST 0
