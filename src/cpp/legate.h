/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
#include <legate/runtime/library.h>
#include <legate/runtime/runtime.h>
#include <legate/runtime/scope.h>
#include <legate/task/exception.h>
#include <legate/task/registrar.h>
#include <legate/task/task.h>
#include <legate/task/task_context.h>
#include <legate/type/type_traits.h>
#include <legate/type/types.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/typedefs.h>
