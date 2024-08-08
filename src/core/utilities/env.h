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

#pragma once

#ifndef LEGATE_CHECK_ENV_VAR_DOCS
#define LEGATE_CHECK_ENV_VAR_DOCS(name) static_assert(true)
#endif

/**
 * @defgroup env Influential Environment Variables in a Legate Program
 */

/**
 * @file
 * @brief Definitions of global environment variables which are understood by Legate.
 * @ingroup env
 */

/**
 * @var LEGATE_TEST
 *
 * @brief Enables "testing" mode in Legate. Possible values: 0, 1.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_TEST);

/**
 * @var LEGATE_SHOW_USAGE
 *
 * @brief Enables verbose resource consumption logging of the base mapper on
 * desctruction. Possible values: 0, 1.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_SHOW_USAGE);

/**
 * @var LEGATE_NEED_CUDA
 *
 * @brief Instructs Legate that it must be CUDA-aware. Possible values: 0, 1.
 *
 * Enabling this, means that Legate must have been configured with CUDA support, and that a
 * CUDA-capable device must be present at startup. If either of these conditions are not met,
 * Legate will abort execution.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_NEED_CUDA);

/**
 * @var LEGATE_NEED_OPENMP
 *
 * @brief Instructs Legate that it must be OpenMP-aware. Possible values: 0, 1.
 *
 * Enabling this, means that Legate must have been configured with OpenMP support, and that a
 * OpenMP-capable device must be present at startup. If either of these conditions are not met,
 * Legate will abort execution.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_NEED_OPENMP);

/**
 * @var LEGATE_NEED_NETWORK
 *
 * @brief Instructs Legate that it must be network-aware. Possible values: 0, 1
 *
 * Enabling this, means that Legate must have been configured with networking support. If
 * either of this condition is not met, Legate will abort execution.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_NEED_NETWORK);

/**
 * @var LEGATE_SHOW_PROGRESS
 *
 * @brief Instructs Legate to emit basic info at that start of each task. Possible values: 0,
 * 1.
 *
 * This variable is useful to visually ensure that a particular task is being called. The
 * progress reports are emitted by Legate before entering into the task body itself.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_SHOW_PROGRESS);

/**
 * @var LEGATE_EMPTY_TASK
 *
 * @brief Instructs Legate to use a dummy empty task body for each task. Possible values: 0, 1.
 *
 * This variable may be enabled to debug logical issues between tasks (for example, control
 * replication issues) by executing the entire task graph without needing to execute the task
 * bodies themselves. This is particularly useful if the task bodies are expensive.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_EMPTY_TASK);

/**
 * @var LEGATE_SYNC_STREAM_VIEW
 *
 * @brief Instructs Legate to synchronize CUDA streams before destruction. Possible values: 0, 1.
 *
 * This variable may be enabled to debug asynchronous issues with CUDA streams. A program which
 * produces different results with this variable enabled and disabled very likely has a race
 * condition between streams. This is especially useful when combined with
 * CUDA_LAUNCH_BLOCKING.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_SYNC_STREAM_VIEW);

/**
 * @var LEGATE_LOG_MAPPING
 *
 * @brief Instructs Legate to emit mapping decisions to stdout. Possible values: 0, 1.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_LOG_MAPPING);

/**
 * @var LEGATE_LOG_PARTITIONING
 *
 * @brief Instructs Legate to emit partitioning decisions to stdout. Possible values: 0, 1.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_LOG_PARTITIONING);

/**
 * @var LEGATE_WARMUP_NCCL
 *
 * @brief Instructs Legate to "warm up" NCCL during startup. Possible values: 0, 1.
 *
 * NCCL usually has a relatively high startup cost the first time any collective communication
 * is performed. This could corrupt performance measurements if that startup is performed in
 * the hot-path.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_WARMUP_NCCL);

/**
 * @var LEGION_DEFAULT_ARGS
 *
 * @brief Default arguments to pass to Legion initialization. Possible values: a string.
 *
 * These arguments are passed verbatim to Legion during runtime startup.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGION_DEFAULT_ARGS);

#undef LEGATE_CHECK_ENV_VAR_DOCS
