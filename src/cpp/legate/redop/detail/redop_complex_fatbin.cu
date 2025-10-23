/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/redop/complex.h>
#include <legate/redop/detail/fatbin_common.cuh>
#include <legate/type/complex.h>
#include <legate/type/half.h>

LEGATE_DEFINE_REDOP_STUBS(legate_sum_redop_complex_half,
                          legate::SumReduction<legate::Complex<legate::Half>>);

// ------------------------------------------------------------------------------------------

LEGATE_DEFINE_REDOP_STUBS(legate_prod_redop_complex_half,
                          legate::ProdReduction<legate::Complex<legate::Half>>);

// ==========================================================================================

LEGATE_DEFINE_REDOP_STUBS(legate_sum_redop_complex_float,
                          legate::SumReduction<legate::Complex<float>>);

// ------------------------------------------------------------------------------------------

LEGATE_DEFINE_REDOP_STUBS(legate_prod_redop_complex_float,
                          legate::ProdReduction<legate::Complex<float>>);

// ==========================================================================================

LEGATE_DEFINE_REDOP_STUBS(legate_sum_redop_complex_double,
                          legate::SumReduction<legate::Complex<double>>);
