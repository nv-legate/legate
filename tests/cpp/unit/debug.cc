/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/debug.h>

#include <legate.h>

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#include <legate/cuda/detail/cuda_driver_api.h>

#endif

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace debug_test {

namespace {

constexpr std::int64_t VAL          = 42;
constexpr std::int32_t TEST_MAX_DIM = 3;

enum class TaskIDs : std::uint8_t {
  CPU_WRITER_TASK = 0,
  GPU_WRITER_TASK = CPU_WRITER_TASK + TEST_MAX_DIM,
};

static_assert(TEST_MAX_DIM <= LEGATE_MAX_DIM);

template <std::int32_t DIM>
class CPUWriterTask : public legate::LegateTask<CPUWriterTask<DIM>> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{
      legate::LocalTaskID{static_cast<std::int32_t>(TaskIDs::CPU_WRITER_TASK) + DIM}};

  static void cpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    for (auto& output : outputs) {
      auto shape = output.shape<DIM>();
      auto acc   = output.data().write_accessor<std::int64_t, DIM>();
      for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
        acc[*it] = VAL;
      }
    }
  }
};

// Add GPUWriterTask to enable testing on both GPU and CPU when the machine has a GPU
template <std::int32_t DIM>
class GPUWriterTask : public legate::LegateTask<GPUWriterTask<DIM>> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{
      legate::LocalTaskID{static_cast<std::int32_t>(TaskIDs::GPU_WRITER_TASK) + DIM}};

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  static void gpu_variant(legate::TaskContext context)
  {
    auto outputs = context.outputs();
    auto&& api   = legate::cuda::detail::get_cuda_driver_api();

    for (auto& output : outputs) {
      auto shape  = output.shape<DIM>();
      auto acc    = output.data().write_accessor<std::int64_t, DIM>();
      auto stream = context.get_task_stream();
      auto vals   = std::vector<std::int64_t>(shape.volume(), VAL);
      auto* ptr   = acc.ptr(shape);

      api->mem_cpy_async(ptr, vals.data(), sizeof(*ptr) * shape.volume(), stream);
    }
  }
#endif
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_debug";
  static void registration_callback(legate::Library library)
  {
    CPUWriterTask<1>::register_variants(library);
    CPUWriterTask<2>::register_variants(library);
    CPUWriterTask<3>::register_variants(library);
    GPUWriterTask<1>::register_variants(library);
    GPUWriterTask<2>::register_variants(library);
    GPUWriterTask<3>::register_variants(library);
  }
};

class DebugUnit : public RegisterOnceFixture<Config> {};

class DebugString : public RegisterOnceFixture<Config>,
                    public ::testing::WithParamInterface<
                      std::tuple<std::int32_t, legate::Shape, std::string_view>> {};

INSTANTIATE_TEST_SUITE_P(
  DebugUnit,
  DebugString,
  ::testing::Values(
    std::make_tuple(1, legate::Shape{3}, "[42, 42, 42]"),
    std::make_tuple(2, legate::Shape{3, 2}, "[[42, 42], [42, 42], [42, 42]]"),
    std::make_tuple(3,
                    legate::Shape{2, 3, 2},
                    "[[[42, 42], [42, 42], [42, 42]], [[42, 42], [42, 42], [42, 42]]]")));

template <std::int32_t DIM, typename TASK>
void test_debug_array(const legate::Shape& array_shape, std::string_view expect_result)
{
  auto runtime = legate::Runtime::get_runtime();
  auto library = runtime->find_library(Config::LIBRARY_NAME);
  auto store   = runtime->create_store(array_shape, legate::int64());
  auto task    = runtime->create_task(library, TASK::TASK_CONFIG.task_id());
  task.add_input(store);
  task.add_output(store);
  runtime->submit(std::move(task));

  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<std::int64_t, DIM>();
  auto shape   = p_store.shape<DIM>();

  auto result = legate::print_dense_array(acc, shape);
  ASSERT_EQ(result, expect_result);
}

class DebugFn {
 public:
  template <std::int32_t DIM>
  void operator()(const legate::Shape& array_shape, std::string_view expect_result)
  {
    test_debug_array<DIM, CPUWriterTask<DIM>>(array_shape, expect_result);
  }
};

class DebugCUDAFn {
 public:
  template <std::int32_t DIM>
  void operator()(const legate::Shape& array_shape, std::string_view expect_result)
  {
    test_debug_array<DIM, GPUWriterTask<DIM>>(array_shape, expect_result);
  }
};

}  // namespace

TEST_P(DebugString, Array)
{
  auto& [dim, shape, expect_result] = GetParam();

  legate::dim_dispatch(dim, DebugFn{}, shape, expect_result);
}

TEST_P(DebugString, CUDAArray)
{
  if (legate::get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    GTEST_SKIP() << "Skip the test if no GPU is found";
  }

  if (!LEGATE_DEFINED(LEGATE_USE_CUDA)) {
    GTEST_SKIP() << "Skip the test if no LEGATE_USE_CUDA is defined";
  }

  auto& [dim, shape, expect_result] = GetParam();

  legate::dim_dispatch(dim, DebugCUDAFn{}, shape, expect_result);
}

}  // namespace debug_test
