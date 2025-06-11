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

#undef LEGATE_CHECK_ENV_VAR_DOCS

/** @} */
