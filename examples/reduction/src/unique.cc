/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <unordered_set>

namespace reduction {

namespace {

template <typename VAL>
void add_to_set(std::unordered_set<VAL>& unique_values, legate::Store input)
{
  auto shape = input.shape<1>();
  if (shape.empty()) {
    return;
  }
  auto acc = input.read_accessor<VAL, 1>();
  for (legate::PointInRectIterator<1> it(shape, false /*fortran_order*/); it.valid(); ++it) {
    unique_values.insert(acc[*it]);
  }
}

template <typename VAL>
void copy_to_output(legate::Store output, const std::unordered_set<VAL>& values)
{
  if (values.empty()) {
    output.bind_empty_data();
    return;
  }

  std::int64_t num_values = values.size();
  auto out_buf =
    output.create_output_buffer<VAL, 1>(legate::Point<1>(num_values), true /*bind_buffer*/);
  std::int64_t idx = 0;
  for (const auto& value : values) {
    out_buf[idx++] = value;
  }
}

template <legate::Type::Code CODE>
constexpr bool is_supported =
  !(legate::is_floating_point<CODE>::value || legate::is_complex<CODE>::value);
struct unique_fn {
  template <legate::Type::Code CODE, std::enable_if_t<is_supported<CODE>>* = nullptr>
  void operator()(legate::Array& output, std::vector<legate::Array>& inputs)
  {
    using VAL = legate::legate_type_of<CODE>;

    std::unordered_set<VAL> unique_values;
    // Find unique values across all inputs
    for (auto& input : inputs) {
      add_to_set(unique_values, input.data());
    }
    // Copy the set of unique values to the output store
    copy_to_output(output.data(), unique_values);
  }

  template <legate::Type::Code CODE, std::enable_if_t<!is_supported<CODE>>* = nullptr>
  void operator()(legate::Array& output, std::vector<legate::Array>& inputs)
  {
    LEGATE_ABORT("Should not be called");
  }
};

}  // namespace

class UniqueTask : public Task<UniqueTask, UNIQUE> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto inputs = context.inputs();
    auto output = context.output(0);
    legate::type_dispatch(output.type().code(), unique_fn{}, output, inputs);
  }
};

}  // namespace reduction

namespace {

static void __attribute__((constructor)) register_tasks()
{
  reduction::UniqueTask::register_variants();
}

}  // namespace
