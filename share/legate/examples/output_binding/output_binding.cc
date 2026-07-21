/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <output_binding.h>

#include <legate.h>

#include <legate/utilities/detail/traced_exception.h>

#include <cstdint>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

constexpr std::int32_t DIM = 1;

// Distinct bases so a check task can tell which producer wrote a store.
constexpr std::int64_t BASE_BOUND_BUFFER = 1000;
constexpr std::int64_t BASE_BIND_DATA    = 2000;
constexpr std::int64_t BASE_TASK_LOCAL   = 3000;

// This rank's index within the producing task's launch (0 for a single/sequential task).
[[nodiscard]] std::uint64_t launch_rank(const legate::TaskContext& context)
{
  const auto& task_index = context.get_task_index();

  return task_index.dim() > 0 ? static_cast<std::uint64_t>(task_index[0]) : 0;
}

[[noreturn]] void fail(const std::string& message)
{
  throw legate::detail::TracedException<std::runtime_error>{message};
}

}  // namespace

void ProduceBoundBufferTask::cpu_variant(legate::TaskContext context)
{
  const auto num  = context.scalar(0).value<std::uint64_t>();
  const auto base = context.scalar(1).value<std::int64_t>();
  auto output     = context.output(0);
  // Global offset of this rank's piece: pieces are concatenated in rank order.
  const auto offset = static_cast<std::int64_t>(launch_rank(context) * num);

  // A temporary scratch buffer that is *not* bound to any store: this exercises the plain
  // create_buffer path (intra-task allocation). We fill it, read it back, and use it below. This
  // is a CPU task, so the buffer is allocated in system memory.
  auto scratch = legate::create_buffer<std::int64_t, DIM>(
    legate::NDPoint<DIM>{static_cast<std::int64_t>(num)}, legate::Memory::SYSTEM_MEM);

  for (std::uint64_t i = 0; i < num; ++i) {
    scratch[legate::NDPoint<DIM>{static_cast<legate::coord_t>(i)}] = static_cast<std::int64_t>(i);
  }

  // Create the output buffer and bind it to the (unbound) output store in a single call.
  auto out_buf = output.create_output_buffer<std::int64_t, DIM>(
    legate::NDPoint<DIM>{static_cast<std::int64_t>(num)}, /*bind_buffer=*/true);

  for (std::uint64_t i = 0; i < num; ++i) {
    const auto p = legate::NDPoint<DIM>{static_cast<legate::coord_t>(i)};

    out_buf[p] = base + offset + scratch[p];
  }

  // The scratch buffer can be released eagerly.
  scratch.destroy();
}

void ProduceViaBindDataTask::cpu_variant(legate::TaskContext context)
{
  const auto num    = context.scalar(0).value<std::uint64_t>();
  const auto base   = context.scalar(1).value<std::int64_t>();
  auto output       = context.output(0);
  const auto offset = static_cast<std::int64_t>(launch_rank(context) * num);

  // Create the buffer without binding, fill it, then bind it in a separate step.
  auto buffer = output.create_output_buffer<std::int64_t, DIM>(
    legate::NDPoint<DIM>{static_cast<std::int64_t>(num)}, /*bind_buffer=*/false);

  for (std::uint64_t i = 0; i < num; ++i) {
    buffer[legate::NDPoint<DIM>{static_cast<legate::coord_t>(i)}] =
      base + offset + static_cast<std::int64_t>(i);
  }

  output.bind_data(buffer, legate::NDPoint<DIM>{static_cast<std::int64_t>(num)});
}

void ProduceViaTaskLocalBufferTask::cpu_variant(legate::TaskContext context)
{
  const auto num    = context.scalar(0).value<std::uint64_t>();
  const auto base   = context.scalar(1).value<std::int64_t>();
  auto output       = context.output(0);
  const auto offset = static_cast<std::int64_t>(launch_rank(context) * num);

  // The type-erased create_output_buffer path returns a TaskLocalBuffer, binding it to the store.
  auto tl_buffer = output.create_output_buffer(legate::Point{static_cast<legate::coord_t>(num)},
                                               /*bind_buffer=*/true);

  // Recover a typed view to write into it.
  auto buffer = static_cast<legate::Buffer<std::int64_t, DIM>>(tl_buffer);

  for (std::uint64_t i = 0; i < num; ++i) {
    buffer[legate::NDPoint<DIM>{static_cast<legate::coord_t>(i)}] =
      base + offset + static_cast<std::int64_t>(i);
  }
}

void CheckTask::cpu_variant(legate::TaskContext context)
{
  const auto base  = context.scalar(0).value<std::int64_t>();
  const auto input = context.input(0);
  const auto shape = input.shape<DIM>();
  auto acc         = input.read_accessor<std::int64_t, DIM>();

  // The accessor is indexed in global coordinates, so the element at global position p must be
  // base + p regardless of which rank produced (or now reads) that position.
  for (auto&& point : shape.iterate()) {
    const auto expected = base + point[0];

    if (acc[point] != expected) {
      std::stringstream ss;

      ss << "output_binding CheckTask (base=" << base << ") mismatch at " << point[0]
         << ": expected " << expected << ", got " << acc[point];
      fail(std::move(ss).str());
    }
  }
}

void ProduceBound2DTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM2 = 2;
  const auto rows             = context.scalar(0).value<std::uint64_t>();
  const auto cols             = context.scalar(1).value<std::uint64_t>();
  const auto base             = context.scalar(2).value<std::int64_t>();
  auto output                 = context.output(0);
  // Pieces are concatenated along the rows (dimension 0).
  const auto row_offset = launch_rank(context) * rows;

  auto buffer = output.create_output_buffer<std::int64_t, DIM2>(
    legate::NDPoint<DIM2>{static_cast<legate::coord_t>(rows), static_cast<legate::coord_t>(cols)},
    /*bind_buffer=*/true);

  for (std::uint64_t i = 0; i < rows; ++i) {
    const auto global_row = row_offset + i;

    for (std::uint64_t j = 0; j < cols; ++j) {
      buffer[legate::NDPoint<DIM2>{static_cast<legate::coord_t>(i),
                                   static_cast<legate::coord_t>(j)}] =
        base + static_cast<std::int64_t>((global_row * cols) + j);
    }
  }
}

void Check2DTask::cpu_variant(legate::TaskContext context)
{
  constexpr std::int32_t DIM2 = 2;
  const auto base             = context.scalar(0).value<std::int64_t>();
  const auto input            = context.input(0);
  const auto shape            = input.shape<DIM2>();
  const auto cols             = shape.hi[1] - shape.lo[1] + 1;
  auto acc                    = input.read_accessor<std::int64_t, DIM2>();

  for (auto&& point : shape.iterate()) {
    const auto expected = base + (point[0] * cols) + point[1];

    if (acc[point] != expected) {
      std::stringstream ss;

      ss << "output_binding Check2DTask (base=" << base << ") mismatch at (" << point[0] << ','
         << point[1] << "): expected " << expected << ", got " << acc[point];
      fail(std::move(ss).str());
    }
  }
}

void CheckWholeTask::cpu_variant(legate::TaskContext context)
{
  const auto base  = context.scalar(0).value<std::int64_t>();
  const auto total = context.scalar(1).value<std::uint64_t>();
  const auto input = context.input(0);
  const auto shape = input.shape<DIM>();
  auto acc         = input.read_accessor<std::int64_t, DIM>();

  if (shape.volume() != total) {
    std::stringstream ss;

    ss << "output_binding CheckWholeTask: gathered store has volume " << shape.volume()
       << ", expected " << total;
    fail(std::move(ss).str());
  }

  for (auto&& point : shape.iterate()) {
    const auto expected = base + point[0];

    if (acc[point] != expected) {
      std::stringstream ss;

      ss << "output_binding CheckWholeTask (base=" << base << ") mismatch at " << point[0]
         << ": expected " << expected << ", got " << acc[point];
      fail(std::move(ss).str());
    }
  }
}

namespace {

// Submits a producer task that binds `store`, followed by a checker task that verifies it with its
// natural partition (each rank checks its own piece).
template <typename ProducerTask>
void produce_and_check(legate::Runtime* runtime,
                       const legate::Library& library,
                       const legate::LogicalStore& store,
                       std::uint64_t num,
                       std::int64_t base)
{
  auto producer = runtime->create_task(library, ProducerTask::TASK_CONFIG.task_id());

  producer.add_output(store);
  producer.add_output(runtime->create_store(legate::Shape{42}, legate::int32()));
  producer.add_scalar_arg(legate::Scalar{num});
  producer.add_scalar_arg(legate::Scalar{base});
  runtime->submit(std::move(producer));

  auto checker = runtime->create_task(library, CheckTask::TASK_CONFIG.task_id());

  checker.add_input(store);
  checker.add_scalar_arg(legate::Scalar{base});
  runtime->submit(std::move(checker));
}

// Reads the whole store replicated on every rank (broadcast), exercising the all-gather out of the
// unbound store's Opaque partition into a NoPartition target.
void check_whole(legate::Runtime* runtime,
                 const legate::Library& library,
                 const legate::LogicalStore& store,
                 std::int64_t base,
                 std::uint64_t total)
{
  auto checker  = runtime->create_task(library, CheckWholeTask::TASK_CONFIG.task_id());
  auto variable = checker.add_input(store);

  checker.add_constraint(legate::broadcast(variable));
  checker.add_scalar_arg(legate::Scalar{base});
  checker.add_scalar_arg(legate::Scalar{total});
  runtime->submit(std::move(checker));
}

void expect_extents(const legate::LogicalStore& store,
                    const std::vector<std::uint64_t>& expected,
                    std::string_view name)
{
  if (store.unbound()) {
    throw legate::detail::TracedException<std::runtime_error>{std::string{name} +
                                                              " is still unbound after production"};
  }

  const auto extents = store.shape().extents().data();

  if (extents != expected) {
    std::stringstream ss;

    ss << name << " has unexpected shape " << store.shape().extents();
    throw legate::detail::TracedException<std::runtime_error>{std::move(ss).str()};
  }
}

void run_example()
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto library        = runtime->create_library("legate_output_binding");

  ProduceBoundBufferTask::register_variants(library);
  ProduceViaBindDataTask::register_variants(library);
  ProduceViaTaskLocalBufferTask::register_variants(library);
  CheckTask::register_variants(library);
  ProduceBound2DTask::register_variants(library);
  Check2DTask::register_variants(library);
  CheckWholeTask::register_variants(library);

  // Per-rank piece size; the global store spans `nranks * NUM` elements.
  constexpr std::uint64_t NUM = 16;
  const auto nranks           = static_cast<std::uint64_t>(runtime->node_count());
  const auto total            = nranks * NUM;

  // Each store is created *unbound*: only its type and dimensionality are known up front. The
  // producer tasks give them a shape by binding a buffer (one piece per rank).
  auto store1 = runtime->create_store(legate::int64(), DIM);
  auto store2 = runtime->create_store(legate::int64(), DIM);
  auto store3 = runtime->create_store(legate::int64(), DIM);

  // base=1000: create_output_buffer(bind=true); base=2000: bind_data; base=3000: TaskLocalBuffer.
  produce_and_check<ProduceBoundBufferTask>(runtime, library, store1, NUM, BASE_BOUND_BUFFER);
  produce_and_check<ProduceViaBindDataTask>(runtime, library, store2, NUM, BASE_BIND_DATA);
  produce_and_check<ProduceViaTaskLocalBufferTask>(runtime, library, store3, NUM, BASE_TASK_LOCAL);

  // Gather each store onto every rank and validate the fully-stitched contents (exercises the
  // Opaque -> NoPartition all-gather copy path).
  check_whole(runtime, library, store1, BASE_BOUND_BUFFER, total);
  check_whole(runtime, library, store2, BASE_BIND_DATA, total);
  check_whole(runtime, library, store3, BASE_TASK_LOCAL, total);

  // A 2D unbound store, to exercise multi-dimensional buffer strides through the binding path.
  // Pieces are concatenated along the rows: the global store is (nranks * ROWS) x COLS.
  constexpr std::uint64_t ROWS   = 4;
  constexpr std::uint64_t COLS   = 5;
  constexpr std::int64_t BASE_2D = 4000;
  auto store2d                   = runtime->create_store(legate::int64(), 2);
  {
    auto producer = runtime->create_task(library, ProduceBound2DTask::TASK_CONFIG.task_id());

    auto part1 = producer.add_output(store2d);
    auto part2 =
      producer.add_output(runtime->create_store(legate::Shape{100, 100}, legate::int64()));
    producer.add_constraint(legate::broadcast(part1, {1}));
    producer.add_constraint(legate::broadcast(part2, {1}));
    producer.add_scalar_arg(legate::Scalar{ROWS});
    producer.add_scalar_arg(legate::Scalar{COLS});
    producer.add_scalar_arg(legate::Scalar{BASE_2D});
    runtime->submit(std::move(producer));

    auto checker = runtime->create_task(library, Check2DTask::TASK_CONFIG.task_id());

    checker.add_input(store2d);
    checker.add_scalar_arg(legate::Scalar{BASE_2D});
    runtime->submit(std::move(checker));
  }

  expect_extents(store1, {total}, "store1");
  expect_extents(store2, {total}, "store2");
  expect_extents(store3, {total}, "store3");
  expect_extents(store2d, {nranks * ROWS, COLS}, "store2d");

  std::cout << "output_binding: all checks passed across " << nranks
            << " rank(s); 1D global shape (" << total << "), 2D global shape (" << (nranks * ROWS)
            << ", " << COLS << ")\n";
}

}  // namespace

int main()
{
  legate::start();

  int result = 0;

  try {
    run_example();
  } catch (const std::exception& e) {
    std::cerr << "output_binding FAILED: " << e.what() << '\n';
    result = 1;
  }

  const auto finish_result = legate::finish();

  return result != 0 ? result : finish_result;
}
