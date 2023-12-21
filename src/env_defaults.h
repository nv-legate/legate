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

#pragma once

#define WINDOW_SIZE_DEFAULT 1
#define WINDOW_SIZE_TEST 1

#define FIELD_REUSE_FRAC_DEFAULT 256
#define FIELD_REUSE_FRAC_TEST 1

#define FIELD_REUSE_FREQ_DEFAULT 32
#define FIELD_REUSE_FREQ_TEST 8

#define DISABLE_MPI_DEFAULT 0
#define DISABLE_MPI_TEST 0

#define CONSENSUS_DEFAULT 0
#define CONSENSUS_TEST 0
