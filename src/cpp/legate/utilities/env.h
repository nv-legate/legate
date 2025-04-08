/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/doxygen.h>

#ifndef LEGATE_CHECK_ENV_VAR_DOCS
#define LEGATE_CHECK_ENV_VAR_DOCS(name) static_assert(true)
#endif

/**
 * @addtogroup env
 * @{
 */

/**
 * @file
 * @brief Definitions of global environment variables which are understood by Legate.
 */

/**
 * @var LEGATE_TEST
 *
 * @brief Enables "testing" mode in Legate. Possible values: 0, 1.
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_TEST);

/**
 * @var LEGATE_SHOW_USAGE
 *
 * @brief Enables verbose resource consumption logging of the base mapper on
 * desctruction. Possible values: 0, 1.
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_SHOW_USAGE);

/**
 * @var LEGATE_AUTO_CONFIG
 *
 * @brief Enables Legate's automatic machine configuration heuristics (turned on by default).
 * Possible values: 0, 1.
 *
 * If enabled, Legate will automatically select an appropriate configuration for the current machine
 * (utilizing most of the available hardware resources, such as CPU cores, RAM and GPUs). If
 * disabled, Legate will use a minimal set of resources. In both cases, further configuration is
 * possible through `LEGATE_CONFIG`.
 *
 * If unset, equivalent to 1 (true).
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_AUTO_CONFIG);

/**
 * @var LEGATE_SHOW_CONFIG
 *
 * @brief Instructs Legate to print out its hardware resource configuration prior to starting.
 * Possible values: 0, 1.
 *
 * This variable can be used to visually confirm that Legate's automatic configuration
 * heuristics are picking up appropriate settings for your machine. If a setting is wrong, you
 * can explicitly override it by passing the corresponding flag in `LEGATE_CONFIG`.
 *
 * If unset, equivalent to 0 (false).
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_SHOW_CONFIG);

/**
 * @var LEGATE_SHOW_PROGRESS
 *
 * @brief Instructs Legate to emit basic info at that start of each task. Possible values: 0,
 * 1.
 *
 * This variable is useful to visually ensure that a particular task is being called. The
 * progress reports are emitted by Legate before entering into the task body itself.
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
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_EMPTY_TASK);

/**
 * @var LEGATE_TEST
 *
 * @brief Enables "testing" mode in Legate. Possible values: 0, 1.
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_TEST);

/**
 * @var LEGATE_LOG_MAPPING
 *
 * @brief Instructs Legate to emit mapping decisions to stdout. Possible values: 0, 1.
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_LOG_MAPPING);

/**
 * @var LEGATE_LOG_PARTITIONING
 *
 * @brief Instructs Legate to emit partitioning decisions to stdout. Possible values: 0, 1.
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
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_WARMUP_NCCL);

/**
 * @var LEGION_DEFAULT_ARGS
 *
 * @brief Default arguments to pass to Legion initialization. Possible values: a string.
 *
 * These arguments are passed verbatim to Legion during runtime startup.
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

/**
 * @var LEGATE_IO_USE_VFD_GDS
 *
 * @brief Whether to enable HDF5 Virtual File Driver (VDS) GPUDirectStorage (GDS) Possible
 * values: 0, 1.
 *
 * This variable, if set, enables the use of GDS with HDF5 files, which may dramatically speed
 * up file storage and extraction. By default, it is off.
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_IO_USE_VFD_GDS);

/**
 * @var LEGATE_MAX_EXCEPTION_SIZE
 *
 * @brief Maximum size in bytes for exceptions that can be raised by tasks.
 * Possible values: a 32-bit integer
 *
 * Legate needs an upper bound on the size of exception that can be raised by a task. By default,
 * the maximum exception size is 4096 bytes.
 */
LEGATE_CHECK_ENV_VAR_DOCS(LEGATE_MAX_EXCEPTION_SIZE);

#undef LEGATE_CHECK_ENV_VAR_DOCS

/** @} */
