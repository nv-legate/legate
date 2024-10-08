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

/**
 * @var LEGATE_MPI_WRAPPER
 *
 * @brief Location of the Legate MPI wrapper shared library to load. Possible values: a string.
 *
 * This variable (if set) must either be the relative or absolute path to the shared library,
 * or simply the name of the shared library. In either case, if set, the shared library must
 * exist either at the specified location, or somewhere in the runtime linker path depending on
 * the form of the path.
 *
 * If the path contains the platform-local "preferred" directory separator ('/' on Unixen, '\'
 * on Windows), the path is considered to be relative or absolute. Otherwise, the path is
 * considered to be the name of the shared library. If the path is relative, it must be
 * relative to the current working directory.
 *
 * For example, if `LEGATE_MPI_WRAPPER="foo.so"` then "foo.so" is considered to be the library
 * name, and will be looked up using the runtime linker path. If it does not exist, or the
 * linker fails to load the library, an error is raised.
 *
 * If `LEGATE_MPI_WRAPPER="../relative/path/to/foo.so"` (on a Unix machine) then it is
 * considered to be a relative path. The path is first resolved (relative to the current
 * working directory), then checked for existence. If the shared library does not exist at the
 * resolved location, an error is raised.
 *
 * If `LEGATE_MPI_WRAPPER="/full/path/to/foo.so"` (on a Unix machine) then it is considered to
 * be an absolute path. The path is checked for existence and if it does not exist, an error is
 * raised.
 *
 * If the variable is not set, the default wrapper shipped with Legate will be used. Depending
 * on the flavor and version of your locally installed MPI, however, this may result in runtime
 * dynamic link errors.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_MPI_WRAPPER);

/**
 * @var LEGATE_CUDA_DRIVER
 *
 * @brief Location of the CUDA driver shared library to load. Possible values: a string.
 *
 * If not set, defaults to "libcuda.so.1", which is looked up using the usual system library
 * path mechanisms. The user should generally not need to set this variable, but it can be
 * useful in case the driver needs to be interposed by a user-supplied shim.
 *
 * @ingroup env
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_CUDA_DRIVER);

/**
 * @var LEGATE_INLINE_TASK_LAUNCH
 *
 * @brief Instructs Legate to launch tasks "inline" whenever possible. Possible values: 0, 1.
 *
 * Normally, when a task is launched, Legate goes through the "Legion calling convention" which
 * involves serialization of all arguments, and packaging up the task such that it may be
 * handed off to Legion for later execution. Crucially, this later execution may happen on
 * another thread, possibly on another node.
 *
 * However, for single-processor runs, this process is both overly costly and largely
 * unnecessary. For example, there is no need to perform any partitioning analysis, as -- by
 * virtue of being single-processor -- the data will be used in full. In such cases it may be
 * profitable to launch the tasks directly on the same processor/thread which submitted them,
 * i.e. "inline".
 *
 * Note that enabling this mode will constrain execution to a single processor, even if more
 * are available.
 *
 * This feature is currently marked experimental, and should not be relied upon. The current
 * implementation is not guaranteed to always be profitable. It may offer dramatic speedup in
 * some circumstances, but it may also lead to large slowdowns in others. Future improvements
 * will seek to improve this, at which point it will be moved to the normal Legate namespace.
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_INLINE_TASK_LAUNCH);

#undef LEGATE_CHECK_ENV_VAR_DOCS
