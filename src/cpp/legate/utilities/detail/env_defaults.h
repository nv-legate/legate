/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// These values are copied manually in legate.settings and there is a Python
// unit test that will maintain that these values and the Python settings
// values agree. If these values are modified, the corresponding Python values
// must also be updated.

#pragma once

#include <legate/utilities/macros.h>

#define LEGATE_MAX_EXCEPTION_SIZE_DEFAULT 4096
#define LEGATE_MAX_EXCEPTION_SIZE_TEST 4096

// 1 << 20 (need actual number for python to parse)
#define LEGATE_MIN_GPU_CHUNK_DEFAULT 1048576
#define LEGATE_MIN_GPU_CHUNK_TEST 2

// 1 << 14 (need actual number for python to parse)
#define LEGATE_MIN_CPU_CHUNK_DEFAULT 16384
#define LEGATE_MIN_CPU_CHUNK_TEST 2

// 1 << 17 (need actual number for python to parse)
#define LEGATE_MIN_OMP_CHUNK_DEFAULT 131072
#define LEGATE_MIN_OMP_CHUNK_TEST 2

// Have a reasonably big window so internal ops wouldn't make the window flush undesirably frequent
#define LEGATE_WINDOW_SIZE_DEFAULT 1
#define LEGATE_WINDOW_SIZE_TEST 1

#define LEGATE_FIELD_REUSE_FRAC_DEFAULT 256
#define LEGATE_FIELD_REUSE_FRAC_TEST 1

#define LEGATE_FIELD_REUSE_FREQ_DEFAULT 32
#define LEGATE_FIELD_REUSE_FREQ_TEST 8

#define LEGATE_DISABLE_MPI_DEFAULT 0
#define LEGATE_DISABLE_MPI_TEST 0

#define LEGATE_CONSENSUS_DEFAULT 0
#define LEGATE_CONSENSUS_TEST 0
