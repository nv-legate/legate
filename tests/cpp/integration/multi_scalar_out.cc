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

#include "legate/utilities/detail/enumerate.h"
#include "legate/utilities/detail/zip.h"

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>
#include <string_view>

namespace multiscalarout {

namespace {

constexpr std::string_view EXN_MSG = "This must be caught on the caller side";

// NOLINTBEGIN(readability-magic-numbers)

class WriteFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::PhysicalStore& store, const legate::Scalar& scalar) const
  {
    using T      = legate::type_of_t<CODE>;
    auto&& shape = store.shape<DIM>();

    store.write_accessor<T, DIM>()[shape.lo] = scalar.value<T>();
  }
};

class ReduceFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::PhysicalStore& store,
                  const legate::Scalar& scalar,
                  const legate::Rect<1>& in_shape) const
  {
    if (in_shape.empty()) {
      return;
    }

    using T          = legate::type_of_t<CODE>;
    auto&& acc       = store.reduce_accessor<legate::SumReduction<T>, true, DIM>();
    auto&& red_shape = store.shape<DIM>();

    for (legate::PointInRectIterator<1> it{in_shape}; it.valid(); ++it) {
      acc[red_shape.lo].reduce(scalar.value<T>());
    }
  }
};

class WriterTask : public legate::LegateTask<WriterTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};

  static void cpu_variant(legate::TaskContext context)
  {
    auto&& outputs = context.outputs();
    auto&& scalars = context.scalars();

    for (auto&& [output, scalar] : legate::detail::zip_equal(outputs, scalars)) {
      double_dispatch(output.dim(), output.type().code(), WriteFn{}, output.data(), scalar);
    }
  }
};

class ReducerTask : public legate::LegateTask<ReducerTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{1};
  static void cpu_variant(legate::TaskContext context)
  {
    auto&& reductions = context.reductions();
    auto&& scalars    = context.scalars();
    auto&& in_shape   = context.input(0).shape<1>();

    for (auto&& [reduction, scalar] : legate::detail::zip_equal(reductions, scalars)) {
      double_dispatch(
        reduction.dim(), reduction.type().code(), ReduceFn{}, reduction.data(), scalar, in_shape);
    }
  }
};

class MixedTask : public legate::LegateTask<MixedTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{2};

  static void cpu_variant(legate::TaskContext context)
  {
    auto&& outputs    = context.outputs();
    auto&& reductions = context.reductions();
    auto&& scalars    = context.scalars();

    auto&& in_shape = context.input(0).shape<1>();

    std::uint32_t out_idx = 0;
    std::uint32_t red_idx = 0;
    for (auto&& [idx, scalar] : legate::detail::enumerate(scalars)) {
      if (idx % 3 == 1) {
        auto&& reduction = reductions[red_idx++];
        double_dispatch(
          reduction.dim(), reduction.type().code(), ReduceFn{}, reduction.data(), scalar, in_shape);
      } else {
        auto&& output = outputs[out_idx++];
        double_dispatch(output.dim(), output.type().code(), WriteFn{}, output.data(), scalar);
      }
    }
  }
};

class UnboundTask : public legate::LegateTask<UnboundTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{3};

  static void cpu_variant(legate::TaskContext context)
  {
    auto&& outputs = context.outputs();
    auto&& scalars = context.scalars();

    for (auto&& [output, scalar] : legate::detail::zip_shortest(outputs, scalars)) {
      double_dispatch(output.dim(), output.type().code(), WriteFn{}, output.data(), scalar);
    }

    outputs.back().data().bind_empty_data();
  }
};

class ExnTask : public legate::LegateTask<ExnTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{4};

  static void cpu_variant(legate::TaskContext /*context*/)
  {
    throw legate::TaskException{std::string{EXN_MSG}};
  }
};

class CheckFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::PhysicalStore& store, const legate::Scalar& scalar) const
  {
    using T      = legate::type_of_t<CODE>;
    auto&& shape = store.shape<DIM>();

    EXPECT_EQ((store.read_accessor<T, DIM>()[shape.lo]), scalar.value<T>());

    auto alloc = store.get_inline_allocation();

    EXPECT_EQ(*static_cast<const T*>(alloc.ptr), scalar.value<T>());
  }
};

class CheckerTask : public legate::LegateTask<CheckerTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{5};

  static void cpu_variant(legate::TaskContext context)
  {
    auto&& output = context.input(0).data();
    auto&& scalar = context.scalar(0);
    double_dispatch(output.dim(), output.type().code(), CheckFn{}, output, scalar);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_multi_scalar";

  static void registration_callback(legate::Library library)
  {
    WriterTask::register_variants(library);
    ReducerTask::register_variants(library);
    MixedTask::register_variants(library);
    UnboundTask::register_variants(library);
    ExnTask::register_variants(library);
    CheckerTask::register_variants(library);
  }
};

class MultiScalarOut : public RegisterOnceFixture<Config> {};

void test_writer_auto(legate::Library library,
                      const std::vector<legate::LogicalStore>& stores,
                      const std::vector<legate::Scalar>& scalars)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, WriterTask::TASK_ID);
  for (auto&& store : stores) {
    task.add_output(store);
  }
  for (auto&& scalar : scalars) {
    task.add_scalar_arg(scalar);
  }
  runtime->submit(std::move(task));
}

void test_reducer_auto(legate::Library library,
                       const legate::LogicalStore& input,
                       const std::vector<legate::LogicalStore>& reductions,
                       const std::vector<legate::Scalar>& scalars)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, ReducerTask::TASK_ID);
  task.add_input(input);
  for (auto&& reduction : reductions) {
    task.add_reduction(reduction, legate::ReductionOpKind::ADD);
  }
  for (auto&& scalar : scalars) {
    task.add_scalar_arg(scalar);
  }
  runtime->submit(std::move(task));
}

void test_reducer_manual(legate::Library library,
                         const legate::LogicalStore& input,
                         const std::vector<legate::LogicalStore>& reductions,
                         const std::vector<legate::Scalar>& scalars)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, ReducerTask::TASK_ID, {2});
  task.add_input(input.partition_by_tiling({3}));
  for (auto&& reduction : reductions) {
    task.add_reduction(reduction, legate::ReductionOpKind::ADD);
  }
  for (auto&& scalar : scalars) {
    task.add_scalar_arg(scalar);
  }
  runtime->submit(std::move(task));
}

void test_mixed_auto(legate::Library library,
                     const legate::LogicalStore& input,
                     const std::vector<legate::LogicalStore>& out_or_reds,
                     const std::vector<legate::Scalar>& scalars)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, MixedTask::TASK_ID);
  task.add_input(input);
  for (auto&& [idx, store] : legate::detail::enumerate(out_or_reds)) {
    if (idx % 3 == 1) {
      task.add_reduction(store, legate::ReductionOpKind::ADD);
    } else {
      task.add_output(store);
    }
  }
  for (auto&& scalar : scalars) {
    task.add_scalar_arg(scalar);
  }
  runtime->submit(std::move(task));
}

void test_unbound(legate::Library library,
                  const std::vector<legate::LogicalStore>& stores,
                  const std::vector<legate::Scalar>& scalars)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, UnboundTask::TASK_ID);
  for (auto&& store : stores) {
    task.add_output(store);
  }
  for (auto&& scalar : scalars) {
    task.add_scalar_arg(scalar);
  }
  task.add_output(runtime->create_store(legate::int64()));
  runtime->submit(std::move(task));
}

void test_exn_and_unbound(legate::Library library,
                          const legate::LogicalStore& input,
                          const legate::LogicalStore& output1,
                          const legate::LogicalStore& output2)
{
  auto runtime = legate::Runtime::get_runtime();
  auto task    = runtime->create_task(library, ExnTask::TASK_ID);
  task.add_input(input);
  task.add_output(output1);
  task.add_output(output2);
  task.throws_exception(true);

  EXPECT_THROW(runtime->submit(std::move(task)), legate::TaskException);
}

void validate_store(const legate::Library& library,
                    const legate::LogicalStore& store,
                    const legate::Scalar& to_match)
{
  auto runtime = legate::Runtime::get_runtime();
  auto p_store = store.get_physical_store();

  double_dispatch(p_store.dim(), p_store.type().code(), CheckFn{}, p_store, to_match);

  auto task = runtime->create_task(library, CheckerTask::TASK_ID);
  task.add_input(store);
  task.add_scalar_arg(to_match);
  runtime->submit(std::move(task));
}

std::vector<legate::Scalar> generate_inputs()
{
  return {
    legate::Scalar{std::int8_t{5}},
    legate::Scalar{std::uint32_t{7}},
    legate::Scalar{std::int64_t{3}},
    legate::Scalar{std::uint16_t{4}},
    legate::Scalar{std::int64_t{2}},
  };
}

std::vector<legate::Scalar> generate_reduction_results()
{
  return {
    legate::Scalar{std::int8_t{25}},
    legate::Scalar{std::uint32_t{35}},
    legate::Scalar{std::int64_t{15}},
    legate::Scalar{std::uint16_t{20}},
    legate::Scalar{std::int64_t{10}},
  };
}

class CreateZeroFn {
 public:
  template <legate::Type::Code CODE>
  legate::Scalar operator()() const
  {
    // clang-tidy bug:
    //
    // error: C-style casts are discouraged; use static_cast
    // [google-readability-casting,-warnings-as-errors]
    // 336 |     return legate::Scalar{legate::type_of_t<CODE>{0}};
    //     |                                             ^
    //     |                                             static_cast<>( )
    //
    // Clearly there is no C-style cast here...
    return legate::Scalar{legate::type_of_t<CODE>{0}};  // NOLINT(google-readability-casting)
  }
};

legate::Scalar create_zero(const legate::Type& type)
{
  return type_dispatch(type.code(), CreateZeroFn{});
}

std::vector<legate::LogicalStore> create_stores(const std::vector<legate::Scalar>& scalars)
{
  auto runtime = legate::Runtime::get_runtime();

  std::vector<legate::LogicalStore> result{};
  result.reserve(scalars.size());

  for (auto&& [idx, scalar] : legate::detail::enumerate(scalars)) {
    auto store = runtime->create_store(legate::Shape{legate::full((idx % 3) + 1, uint64_t{1})},
                                       scalar.type(),
                                       true /*optimize_scalar*/);
    runtime->issue_fill(store, create_zero(scalar.type()));
    result.push_back(std::move(store));
  }

  return result;
}

}  // namespace

TEST_F(MultiScalarOut, WriteAuto)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto values_to_write = generate_inputs();
  auto stores          = create_stores(values_to_write);

  test_writer_auto(library, stores, values_to_write);
  for (auto&& [store, to_match] : legate::detail::zip_equal(stores, values_to_write)) {
    validate_store(library, store, to_match);
  }
}

TEST_F(MultiScalarOut, ReduceAuto)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto input = runtime->create_store(legate::Shape{5}, legate::int64());
  runtime->issue_fill(input, create_zero(input.type()));

  auto values_to_write   = generate_inputs();
  auto reductions        = create_stores(values_to_write);
  auto reduction_results = generate_reduction_results();

  test_reducer_auto(library, input, reductions, values_to_write);
  for (auto&& [reduction, to_match] : legate::detail::zip_equal(reductions, reduction_results)) {
    validate_store(library, reduction, to_match);
  }
}

TEST_F(MultiScalarOut, ReduceManual)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto input = runtime->create_store(legate::Shape{5}, legate::int64());
  runtime->issue_fill(input, create_zero(input.type()));

  auto values_to_write   = generate_inputs();
  auto reductions        = create_stores(values_to_write);
  auto reduction_results = generate_reduction_results();

  test_reducer_manual(library, input, reductions, values_to_write);
  for (auto&& [reduction, to_match] : legate::detail::zip_equal(reductions, reduction_results)) {
    validate_store(library, reduction, to_match);
  }
}

TEST_F(MultiScalarOut, MixedAuto)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto input = runtime->create_store(legate::Shape{5}, legate::int64());
  runtime->issue_fill(input, create_zero(input.type()));

  auto values_to_write   = generate_inputs();
  auto out_or_reds       = create_stores(values_to_write);
  auto reduction_results = generate_reduction_results();

  test_mixed_auto(library, input, out_or_reds, values_to_write);

  for (auto&& [idx, store] : legate::detail::enumerate(out_or_reds)) {
    validate_store(library, store, (idx % 3 == 1) ? reduction_results[idx] : values_to_write[idx]);
  }
}

TEST_F(MultiScalarOut, Unbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto values_to_write = generate_inputs();
  auto stores          = create_stores(values_to_write);

  test_unbound(library, stores, values_to_write);

  for (auto&& [store, to_match] : legate::detail::zip_equal(stores, values_to_write)) {
    validate_store(library, store, to_match);
  }
}

TEST_F(MultiScalarOut, ExnAndUnbound)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);

  auto input = runtime->create_store(legate::Shape{5}, legate::int64());
  runtime->issue_fill(input, create_zero(input.type()));

  auto out1 = runtime->create_store(legate::int64());
  auto out2 = runtime->create_store({1, 1, 1}, legate::int8());

  test_exn_and_unbound(library, input, out1, out2);
}

// NOLINTEND(readability-magic-numbers)

}  // namespace multiscalarout
