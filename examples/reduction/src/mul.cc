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

#include "core/utilities/dispatch.h"

#include "legate_library.h"
#include "reduction_cffi.h"

namespace reduction {

namespace {

struct mul_fn {
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(legate::Store& lhs, legate::Store& rhs1, legate::Store& rhs2)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = lhs.shape<DIM>();

    if (shape.empty()) {
      return;
    }

    auto rhs1_acc = rhs1.read_accessor<VAL, DIM>();
    auto rhs2_acc = rhs2.read_accessor<VAL, DIM>();
    auto lhs_acc  = lhs.write_accessor<VAL, DIM>();

    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto p     = *it;
      lhs_acc[p] = rhs1_acc[p] * rhs2_acc[p];
    }
  }
};

}  // namespace

class MultiplyTask : public Task<MultiplyTask, MUL> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto rhs1 = context.input(0).data();
    auto rhs2 = context.input(1).data();
    auto lhs  = context.output(0).data();

    legate::double_dispatch(lhs.dim(), lhs.code(), mul_fn{}, lhs, rhs1, rhs2);
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::MultiplyTask::register_variants();
}

}  // namespace
