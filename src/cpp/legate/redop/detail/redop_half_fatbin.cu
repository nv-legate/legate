/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/redop/detail/fatbin_common.cuh>
#include <legate/redop/half.h>
#include <legate/type/half.h>

LEGATE_DEFINE_REDOP_STUBS(legate_sum_redop_half, legate::SumReduction<legate::Half>);

// ==========================================================================================

LEGATE_DEFINE_REDOP_STUBS(legate_prod_redop_half, legate::ProdReduction<legate::Half>);

// ==========================================================================================

LEGATE_DEFINE_REDOP_STUBS(legate_max_redop_half, legate::MaxReduction<legate::Half>);

// ==========================================================================================

LEGATE_DEFINE_REDOP_STUBS(legate_min_redop_half, legate::MinReduction<legate::Half>);
