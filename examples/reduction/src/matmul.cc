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

#include "legate_library.h"
#include "reduction_cffi.h"

#include "core/utilities/dispatch.h"

namespace reduction {

namespace {

struct matmul_fn {
  template <legate::Type::Code CODE>
  void operator()(legate::Store& lhs, legate::Store& rhs1, legate::Store& rhs2)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = rhs1.shape<3>().intersection(rhs2.shape<3>());

    if (shape.empty()) return;

    auto rhs1_acc = rhs1.read_accessor<VAL, 3>();
    auto rhs2_acc = rhs2.read_accessor<VAL, 3>();
    auto lhs_acc  = lhs.reduce_accessor<legate::SumReduction<VAL>, true, 3>();

    for (legate::PointInRectIterator<3> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto p = *it;
      lhs_acc.reduce(p, rhs1_acc[p] * rhs2_acc[p]);
    }
  }
};

}  // namespace

class MatMulTask : public Task<MatMulTask, MATMUL> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto rhs1 = context.input(0).data();
    auto rhs2 = context.input(1).data();
    auto lhs  = context.reduction(0).data();

    legate::type_dispatch(lhs.code(), matmul_fn{}, lhs, rhs1, rhs2);
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::MatMulTask::register_variants();
}

}  // namespace
