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

#pragma once

/**
 * @mainpage Legate C++ API reference
 *
 * This is an API reference for Legate's C++ components.
 */

#include "legion.h"
// legion.h has to go before these
#include "core/data/allocator.h"
#include "core/data/logical_store.h"
#include "core/data/scalar.h"
#include "core/data/store.h"
#include "core/legate_c.h"
#include "core/mapping/mapping.h"
#include "core/mapping/operation.h"
#include "core/operation/task.h"
#include "core/partitioning/constraint.h"
#include "core/partitioning/partition.h"
#include "core/runtime/library.h"
#include "core/runtime/runtime.h"
#include "core/runtime/tracker.h"
#include "core/task/exception.h"
#include "core/task/registrar.h"
#include "core/task/task.h"
#include "core/task/task_context.h"
#include "core/type/type_traits.h"
#include "core/utilities/dispatch.h"
#include "core/utilities/typedefs.h"
#include "legate_defines.h"
