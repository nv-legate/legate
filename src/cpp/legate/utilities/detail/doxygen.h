/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @defgroup io Input/Output facilities
 *
 * @brief Utilities for serializing and deserializing Legate stores to disk.
 */

/**
 * @defgroup io-hdf5 HDF5
 * @ingroup io
 *
 * @brief I/O operations backed by HDF5.
 */

/**
 * @defgroup io-kvikio KVikIO
 * @ingroup io
 *
 * @brief I/O operations backed by KVikIO.
 */

/**
 * @defgroup geometry Geometry types
 *
 * @brief Geometry types
 */

/**
 * @defgroup accessor Accessor types
 *
 * Accessors provide an interface to access values in stores. Access modes are encoded
 * in the accessor types so that the compiler can catch invalid accesses. Accessors also
 * provide bounds checks (which can be turned on with a compile flag).
 *
 * All accessors have a `ptr` method that returns a raw pointer to the underlying allocation.
 * The caller can optionally pass an array to query strides of dimensions, necessary for correct
 * accesse. Unlike the accesses mediated by accessors, raw pointer accesses are not protected by
 * Legate, and thus the developer should make sure of safety of the accesses.
 *
 * The most common mistake with raw pointers from reduction accessors are that the code overwrites
 * values to the elements, instead of reducing them. The key contract with reduction is that
 * the values must be reduced to the elements in the store. So, any client code that uses a raw
 * pointer to a reduction store should make sure that it makes updates to the effect of reducing
 * its contributions to the original elements. Not abiding by this contract can lead to
 * non-deterministic conrrectness issues.
 */

/**
 * @defgroup iterator Iterator types
 *
 * @brief Iterator types
 */

/**
 * @defgroup machine Machine objects
 *
 * @brief Objects for interacting with the Machine
 */

/**
 * @defgroup reduction Built-in reduction operators
 *
 * All built-in operators are defined for signed and unsigned integer types. Floating point
 * types (`__half`, `float`, and `double`) are supported by all but bitwise operators. Arithmetic
 * operators also cover complex types `complex<__half>` and `complex<float>`.
 *
 * For details about reduction operators, See Library::register_reduction_operator.
 */

/**
 * @defgroup util Utilities
 *
 * @brief General utilities
 */

/**
 * @defgroup env Influential Environment Variables in a Legate Program
 * @ingroup util
 */

/**
 * @defgroup types Type system
 *
 * @brief Objects for the specification and management of types
 */

/**
 * @defgroup data Data abstractions and allocators
 */

/**
 * @defgroup partitioning Partitioning
 */

/**
 * @defgroup runtime Runtime and library contexts
 *
 * @brief Runtime and Library contexts for the management and launching of tasks
 */

/**
 * @defgroup mapping Mapping
 *
 * @brief Classes and utilities to control the placement and allocation of tasks on the machine
 */

/**
 * @defgroup task Task
 *
 * @brief Classes and utilities to define tasks
 */

/**
 * @defgroup tuning Tuning
 *
 * @brief Classes and utilities to define tuning inside a Scope
 */
