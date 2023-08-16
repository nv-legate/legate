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

struct reduction_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::Store& ouptut, legate::Store& input)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = input.shape<DIM>();

    if (shape.empty()) return;

    auto in_acc  = input.read_accessor<VAL, DIM>();
    auto red_acc = ouptut.reduce_accessor<legate::SumReduction<VAL>, true, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto p = *it;
      // Coordinates of the contracting dimension are ignored by red_acc via an affine
      // transformation. For example, if the store was 3D and the second dimension was contracted,
      // each point p will go through the following affine transformation to recover the point in
      // the domain prior to the promotion:
      //
      //     | 1  0  0 |     | x |
      //     |         |  *  | y |
      //     | 0  0  1 |     | z |
      //
      // where the "*" operator denotes a matrix-vector multiplication.
      red_acc.reduce(p, in_acc[p]);
    }
  }
};

}  // namespace

class SumOverAxisTask : public Task<SumOverAxisTask, SUM_OVER_AXIS> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto input  = context.input(0).data();
    auto ouptut = context.reduction(0).data();

    legate::double_dispatch(input.dim(), input.code(), reduction_fn{}, ouptut, input);
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::SumOverAxisTask::register_variants();
}

}  // namespace
