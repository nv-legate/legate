/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/**
 * @mainpage Legate C++ API reference
 *
 * This is an API reference for Legate's C++ components.
 */

#include <legion.h>
// legion.h has to go before these
#include <legate_defines.h>
//
#include <legate/data/allocator.h>
#include <legate/data/external_allocation.h>
#include <legate/data/logical_store.h>
#include <legate/data/physical_store.h>
#include <legate/data/scalar.h>
#include <legate/mapping/mapping.h>
#include <legate/mapping/operation.h>
#include <legate/operation/projection.h>
#include <legate/operation/task.h>
#include <legate/partitioning/constraint.h>
#include <legate/partitioning/proxy.h>
#include <legate/runtime/library.h>
#include <legate/runtime/runtime.h>
#include <legate/task/exception.h>
#include <legate/task/registrar.h>
#include <legate/task/task.h>
#include <legate/task/task_config.h>
#include <legate/task/task_context.h>
#include <legate/task/task_signature.h>
#include <legate/task/variant_options.h>
#include <legate/tuning/parallel_policy.h>
#include <legate/tuning/scope.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/typedefs.h>
