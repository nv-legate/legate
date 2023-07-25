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

struct categorize_fn {
  template <legate::Type::Code CODE, std::enable_if_t<!legate::is_complex<CODE>::value>* = nullptr>
  void operator()(legate::Store& result, legate::Store& input, legate::Store& bins)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto in_shape  = result.shape<1>();
    auto bin_shape = bins.shape<1>();

    assert(!bin_shape.empty());
    if (in_shape.empty()) return;

    auto num_bins = bin_shape.hi[0] - bin_shape.lo[0];

    auto in_acc  = input.read_accessor<VAL, 1>();
    auto bin_acc = bins.read_accessor<VAL, 1>();
    auto res_acc = result.write_accessor<uint64_t, 1>();

    for (legate::PointInRectIterator<1> it(in_shape); it.valid(); ++it) {
      auto p      = *it;
      auto& value = in_acc[p];
      for (auto bin_idx = 0; bin_idx < num_bins; ++bin_idx) {
        if (bin_acc[bin_idx] <= value && value < bin_acc[bin_idx + 1]) {
          res_acc[p] = static_cast<uint64_t>(bin_idx);
          break;
        }
      }
    }
  }

  template <legate::Type::Code CODE, std::enable_if_t<legate::is_complex<CODE>::value>* = nullptr>
  void operator()(legate::Store& result, legate::Store& input, legate::Store& bins)
  {
    assert(false);
  }
};

}  // namespace

class CategorizeTask : public Task<CategorizeTask, CATEGORIZE> {
 public:
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& input  = context.inputs().at(0);
    auto& bins   = context.inputs().at(1);
    auto& result = context.outputs().at(0);

    legate::type_dispatch(input.code(), categorize_fn{}, result, input, bins);
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::CategorizeTask::register_variants();
}

}  // namespace
