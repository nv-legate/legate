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

// NOLINTBEGIN(bugprone-reserved-identifier)
#ifndef __LEGATE_C_H__
#define __LEGATE_C_H__
// NOLINTEND(bugprone-reserved-identifier)

#ifndef LEGATE_USE_PYTHON_CFFI
#include "legion/legion_config.h"
//
#include <cstdint>
#endif

// NOLINTBEGIN(modernize-use-using)
typedef enum legate_core_task_id_t {
  LEGATE_CORE_TOPLEVEL_TASK_ID,
  LEGATE_CORE_EXTRACT_SCALAR_TASK_ID,
  LEGATE_CORE_INIT_NCCL_ID_TASK_ID,
  LEGATE_CORE_INIT_NCCL_TASK_ID,
  LEGATE_CORE_FINALIZE_NCCL_TASK_ID,
  LEGATE_CORE_INIT_CPUCOLL_MAPPING_TASK_ID,
  LEGATE_CORE_INIT_CPUCOLL_TASK_ID,
  LEGATE_CORE_FINALIZE_CPUCOLL_TASK_ID,
  LEGATE_CORE_FIXUP_RANGES,
  LEGATE_CORE_OFFSETS_TO_RANGES,
  LEGATE_CORE_RANGES_TO_OFFSETS,
  LEGATE_CORE_FIRST_DYNAMIC_TASK_ID,
  // Legate core runtime allocates LEGATE_CORE_MAX_TASK_ID tasks from Legion upfront. All ID's
  // prior to LEGATE_CORE_FIRST_DYNAMIC_TASK_ID are for specific, bespoke
  // tasks. LEGATE_CORE_MAX_TASK_ID - LEGATE_CORE_FIRST_DYNAMIC_TASK_ID are for "dynamic"
  // tasks, e.g. those created from Python or elsewhere. Hence we make LEGATE_CORE_MAX_TASK_ID
  // large enough so that theres enough slack
  LEGATE_CORE_MAX_TASK_ID = 512,  // must be last
} legate_core_task_id_t;

typedef enum legate_core_proj_id_t {
  // local id 0 always maps to the identity projection (global id 0)
  LEGATE_CORE_DELINEARIZE_PROJ_ID      = 2,
  LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR_ID = 10,
  LEGATE_CORE_MAX_FUNCTOR_ID           = 3000000,
} legate_core_proj_id_t;

typedef enum legate_core_shard_id_t {
  LEGATE_CORE_TOPLEVEL_TASK_SHARD_ID = 0,
  LEGATE_CORE_LINEARIZE_SHARD_ID     = 1,
  // All sharding functors starting from LEGATE_CORE_FIRST_DYNAMIC_FUNCTOR should match the
  // projection functor of the same id. The sharding functor limit is thus the same as the
  // projection functor limit.
} legate_core_shard_id_t;

typedef enum legate_core_tunable_t {
  LEGATE_CORE_TUNABLE_TOTAL_CPUS = 12345,
  LEGATE_CORE_TUNABLE_TOTAL_OMPS,
  LEGATE_CORE_TUNABLE_TOTAL_GPUS,
  LEGATE_CORE_TUNABLE_NUM_NODES,
  LEGATE_CORE_TUNABLE_HAS_SOCKET_MEM,
  LEGATE_CORE_TUNABLE_FIELD_REUSE_SIZE,
} legate_core_tunable_t;

typedef enum legate_core_variant_t {
  LEGATE_NO_VARIANT = 0,
  LEGATE_CPU_VARIANT,
  LEGATE_GPU_VARIANT,
  LEGATE_OMP_VARIANT,
} legate_core_variant_t;

// Match these to numpy_field_type_offsets in legate/numpy/config.py
typedef enum legate_core_type_code_t {
  // Buil-in primitive types
  BOOL_LT       = LEGION_TYPE_BOOL,
  INT8_LT       = LEGION_TYPE_INT8,
  INT16_LT      = LEGION_TYPE_INT16,
  INT32_LT      = LEGION_TYPE_INT32,
  INT64_LT      = LEGION_TYPE_INT64,
  UINT8_LT      = LEGION_TYPE_UINT8,
  UINT16_LT     = LEGION_TYPE_UINT16,
  UINT32_LT     = LEGION_TYPE_UINT32,
  UINT64_LT     = LEGION_TYPE_UINT64,
  FLOAT16_LT    = LEGION_TYPE_FLOAT16,
  FLOAT32_LT    = LEGION_TYPE_FLOAT32,
  FLOAT64_LT    = LEGION_TYPE_FLOAT64,
  COMPLEX64_LT  = LEGION_TYPE_COMPLEX64,
  COMPLEX128_LT = LEGION_TYPE_COMPLEX128,
  // Null type
  NULL_LT,
  // Opaque binary type
  BINARY_LT,
  // Compound types
  FIXED_ARRAY_LT,
  STRUCT_LT,
  // Variable size types
  STRING_LT,
  LIST_LT,
} legate_core_type_code_t;

typedef enum legate_core_reduction_op_kind_t {
  ADD_LT = LEGION_REDOP_KIND_SUM,
  SUB_LT = LEGION_REDOP_KIND_DIFF,
  MUL_LT = LEGION_REDOP_KIND_PROD,
  DIV_LT = LEGION_REDOP_KIND_DIV,
  MAX_LT = LEGION_REDOP_KIND_MAX,
  MIN_LT = LEGION_REDOP_KIND_MIN,
  OR_LT  = LEGION_REDOP_KIND_OR,
  AND_LT = LEGION_REDOP_KIND_AND,
  XOR_LT = LEGION_REDOP_KIND_XOR,
} legate_core_reduction_op_kind_t;

typedef enum legate_core_transform_t {
  LEGATE_CORE_TRANSFORM_SHIFT = 100,
  LEGATE_CORE_TRANSFORM_PROMOTE,
  LEGATE_CORE_TRANSFORM_PROJECT,
  LEGATE_CORE_TRANSFORM_TRANSPOSE,
  LEGATE_CORE_TRANSFORM_DELINEARIZE,
} legate_core_transform_t;

typedef enum legate_core_mapping_tag_t {
  LEGATE_CORE_KEY_STORE_TAG              = 1,
  LEGATE_CORE_MANUAL_PARALLEL_LAUNCH_TAG = 2,
  LEGATE_CORE_TREE_REDUCE_TAG            = 3,
  LEGATE_CORE_JOIN_EXCEPTION_TAG         = 4,
} legate_core_mapping_tag_t;

typedef enum legate_core_reduction_op_id_t {
  LEGATE_CORE_JOIN_EXCEPTION_OP   = 0,
  LEGATE_CORE_MAX_REDUCTION_OP_ID = 1,
} legate_core_reduction_op_id_t;
// NOLINTEND(modernize-use-using)

#ifdef __cplusplus
extern "C" {
#endif

void legate_core_perform_registration(void);

#ifdef __cplusplus
}
#endif

#endif  // __LEGATE_C_H__
