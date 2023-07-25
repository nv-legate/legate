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

#include "core/data/detail/logical_store.h"
#include "legate.h"

namespace {

constexpr int32_t TEST_MAX_DIM = 3;

// Instantiate only the cases exercised in copy tests
template <typename Functor, typename... Fnargs>
constexpr decltype(auto) type_dispatch_for_test(legate::Type::Code code,
                                                Functor f,
                                                Fnargs&&... args)
{
  switch (code) {
    case legate::Type::Code::INT64: {
      return f.template operator()<legate::Type::Code::INT64>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::UINT32: {
      return f.template operator()<legate::Type::Code::UINT32>(std::forward<Fnargs>(args)...);
    }
    case legate::Type::Code::FLOAT64: {
      return f.template operator()<legate::Type::Code::FLOAT64>(std::forward<Fnargs>(args)...);
    }
    default: break;
  }
  assert(false);
  return f.template operator()<legate::Type::Code::INT64>(std::forward<Fnargs>(args)...);
}

enum TaskIDs {
  FILL_TASK          = 0,
  FILL_INDIRECT_TASK = FILL_TASK + TEST_MAX_DIM,
};

template <int32_t DIM>
struct FillTask : public legate::LegateTask<FillTask<DIM>> {
  struct FillTaskBody {
    template <legate::Type::Code CODE>
    void operator()(legate::Store& output, legate::Rect<DIM>& shape, legate::Scalar& seed)
    {
      using VAL    = legate::legate_type_of<CODE>;
      auto acc     = output.write_accessor<VAL, DIM>(shape);
      auto to_fill = seed.value<VAL>();
      size_t i     = 1;
      for (legate::PointInRectIterator<DIM> it(shape); it.valid(); ++it, ++i)
        acc[*it] = i * to_fill;
    }
  };

  static const int32_t TASK_ID = FILL_TASK + DIM;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& output = context.outputs().at(0);
    auto shape   = output.shape<DIM>();
    auto& seed   = context.scalars().at(0);

    if (shape.empty()) return;

    type_dispatch_for_test(output.type().code(), FillTaskBody{}, output, shape, seed);
  }
};

template <int32_t DIM>
legate::Point<DIM> delinearize(size_t index,
                               const legate::Point<DIM>& extents,
                               const legate::Point<DIM>& lo)
{
  legate::Point<DIM> result;
  for (int32_t dim = 0; dim < DIM; ++dim) {
    result[dim] = index % extents[dim];
    index       = index / extents[dim];
  }
  return result + lo;
}

template <int32_t IND_DIM, int32_t DATA_DIM>
struct FillIndirectTask : public legate::LegateTask<FillIndirectTask<IND_DIM, DATA_DIM>> {
  static const int32_t TASK_ID = FILL_INDIRECT_TASK + IND_DIM * TEST_MAX_DIM + DATA_DIM;
  static void cpu_variant(legate::TaskContext& context)
  {
    auto& output    = context.outputs().at(0);
    auto ind_shape  = output.shape<IND_DIM>();
    auto data_shape = context.scalars().at(0).value<legate::Rect<DATA_DIM>>();

    size_t data_vol = data_shape.volume();
    size_t ind_vol  = ind_shape.volume();

    if (0 == ind_vol) return;

    auto data_extents = data_shape.hi - data_shape.lo + legate::Point<DATA_DIM>::ONES();
    auto acc          = output.write_accessor<legate::Point<DATA_DIM>, IND_DIM>(ind_shape);
    auto idx          = 0;
    for (legate::PointInRectIterator<IND_DIM> it(ind_shape); it.valid(); ++it) {
      auto p   = delinearize((data_vol * idx++ / ind_vol) % data_vol, data_extents, data_shape.lo);
      acc[*it] = p;
    }
  }
};

void fill_input(legate::Library library,
                const legate::LogicalStore& src,
                const legate::Scalar& value)
{
  auto runtime = legate::Runtime::get_runtime();
  auto machine = runtime->get_machine();
  auto task    = runtime->create_task(library, FILL_TASK + src.dim());
  task.add_output(src, task.declare_partition());
  task.add_scalar_arg(value);
  runtime->submit(std::move(task));
}

void fill_indirect(legate::Library library,
                   const legate::LogicalStore& ind,
                   const legate::LogicalStore& data)
{
  auto runtime    = legate::Runtime::get_runtime();
  auto machine    = runtime->get_machine();
  int32_t task_id = FILL_INDIRECT_TASK + ind.dim() * TEST_MAX_DIM + data.dim();
  auto task       = runtime->create_task(library, task_id);
  auto part       = task.declare_partition();
  task.add_output(ind, part);
  // Technically indirection fields for gather coipes can have repeated points
  // and thus be initialized in parallel, but we always serialize the
  // initialization to simplify the logic
  task.add_constraint(legate::broadcast(part, legate::from_range(ind.dim())));
  auto domain = legate::to_domain(data.extents());
  switch (data.dim()) {
    case 1: {
      task.add_scalar_arg(legate::Rect<1>(domain));
      break;
    }
    case 2: {
      task.add_scalar_arg(legate::Rect<2>(domain));
      break;
    }
    case 3: {
      task.add_scalar_arg(legate::Rect<3>(domain));
      break;
    }
    default: {
      assert(false);
      break;
    }
  }
  runtime->submit(std::move(task));
  // Important to reset the indirection field's key partition. Here's why: multi-dimensional point
  // types have bigger sizes than the value types used in copy tests, the partitioner will favor the
  // indirection field's key partition over others. Since indirection fields are initialized by
  // sequential tasks in this function, they would end up always serializing downstream indirect
  // copies, which is undesirable.
  ind.impl()->reset_key_partition();
}

std::string to_string(const std::vector<size_t>& shape)
{
  std::stringstream ss;
  ss << "(";
  for (auto& ext : shape) ss << ext << ",";
  ss << ")";
  return std::move(ss).str();
}

}  // namespace
