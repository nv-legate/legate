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

#include "legate/data/detail/logical_store.h"
#include "legate/data/external_allocation.h"
#include "legate/mapping/mapping.h"

#include "legate.h"
#include "utilities/utilities.h"

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
#include "legate/cuda/cuda.h"

#include <cuda_runtime.h>
#endif

#include <cstring>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace index_attach {

#define CHECK_CUDA(...)                                                               \
  do {                                                                                \
    const cudaError_t __result__ = __VA_ARGS__;                                       \
    if (__result__ != cudaSuccess) {                                                  \
      throw std::runtime_error{                                                       \
        fmt::format("Internal CUDA failure with error {} ({}) in file {} at line {}", \
                    cudaGetErrorString(__result__),                                   \
                    cudaGetErrorName(__result__),                                     \
                    __FILE__,                                                         \
                    __LINE__)};                                                       \
    }                                                                                 \
  } while (0)

// NOLINTBEGIN(readability-magic-numbers)

namespace {

constexpr std::size_t TILE_SIZE    = 5;
constexpr std::uint64_t INIT_VALUE = 10;

class AccessStoreFn {
 public:
  template <legate::Type::Code CODE>
  void operator()(legate::TaskContext context)
  {
    using T                    = legate::type_of_t<CODE>;
    auto p_store               = context.input(0).data();
    constexpr std::int32_t DIM = 1;
    ASSERT_EQ(p_store.dim(), DIM);
    auto shape = p_store.shape<DIM>();

    if (shape.empty()) {
      return;
    }

    auto value  = context.scalar(0).value<T>();
    auto rw_acc = p_store.read_write_accessor<T, DIM>();

    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      ASSERT_EQ(rw_acc[*it], static_cast<T>(value));
      rw_acc[*it] = static_cast<T>(INIT_VALUE);
      ASSERT_EQ(rw_acc[*it], static_cast<T>(INIT_VALUE));
    }
  }
};

class AccessTask : public legate::LegateTask<AccessTask> {
 public:
  static constexpr auto TASK_ID = legate::LocalTaskID{0};
  static void cpu_variant(legate::TaskContext context);
};

/*static*/ void AccessTask::cpu_variant(legate::TaskContext context)
{
  auto p_store = context.input(0).data();
  legate::type_dispatch(p_store.code(), AccessStoreFn{}, context);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "legate.external_allocation";
  static void registration_callback(legate::Library library)
  {
    AccessTask::register_variants(library);
  }
};

class IndexAttach : public RegisterOnceFixture<Config> {};

template <typename T>
void test_access_by_task(legate::ExternalAllocation& ext, T value)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(Config::LIBRARY_NAME);
  auto task          = runtime->create_task(context, AccessTask::TASK_ID);
  auto logical_store = runtime->create_store(
    legate::Shape{TILE_SIZE}, legate::primitive_type(legate::type_code_of_v<T>), ext);

  task.add_input(logical_store);
  task.add_output(logical_store);
  task.add_scalar_arg(value);
  runtime->submit(std::move(task));

  logical_store.detach();
}

template <typename T>
void test_sysmem(void* ptr,
                 T value,
                 std::size_t bytes,
                 bool read_only                                             = true,
                 std::optional<legate::ExternalAllocation::Deleter> deleter = std::nullopt)
{
  auto ext_alloc =
    legate::ExternalAllocation::create_sysmem(ptr, bytes, read_only, std::move(deleter));

  ASSERT_EQ(ext_alloc.size(), bytes);
  ASSERT_EQ(ext_alloc.read_only(), read_only);
  ASSERT_EQ(ext_alloc.ptr(), ptr);
  ASSERT_EQ(ext_alloc.target(), legate::mapping::StoreTarget::SYSMEM);
  test_access_by_task<T>(ext_alloc, value);
}

template <typename T>
void do_test(T value)
{
  std::vector<T> alloc(TILE_SIZE, value);
  constexpr std::size_t BYTES = TILE_SIZE * sizeof(T);

  test_sysmem<T>(alloc.data(), value, BYTES, false);
}

void test_gpu_mutuable_access(legate::mapping::StoreTarget store_target)
{
  if (legate::get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    static_cast<void>(store_target);
    return;
  }

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  constexpr std::size_t BYTES = TILE_SIZE * sizeof(std::uint64_t);
  std::vector<std::uint64_t> h_alloc(TILE_SIZE, INIT_VALUE);
  void* d_alloc = nullptr;
  auto deleter  = [](void* ptr) noexcept {
    auto h_buffer      = std::make_unique<std::uint64_t[]>(BYTES);
    void* raw_h_buffer = static_cast<void*>(h_buffer.get());

    ASSERT_NE(raw_h_buffer, nullptr);
    try {
      CHECK_CUDA(cudaMemcpy(raw_h_buffer, ptr, BYTES, cudaMemcpyDeviceToHost));
      // TODO(issue 464)
      // ASSERT_EQ(*(static_cast<std::uint64_t*>(raw_h_buffer)), INIT_VALUE - 1);
      CHECK_CUDA(cudaFree(ptr));
    } catch (const std::exception& e) {
      LEGATE_ABORT(e.what());
    }
  };

  CHECK_CUDA(cudaMalloc(&d_alloc, BYTES));
  CHECK_CUDA(cudaMemcpy(d_alloc, h_alloc.data(), BYTES, cudaMemcpyHostToDevice));

  legate::ExternalAllocation ext_alloc;
  switch (store_target) {
    case legate::mapping::StoreTarget::FBMEM: {
      ext_alloc =
        legate::ExternalAllocation::create_fbmem(0, d_alloc, BYTES, false, std::move(deleter));
      break;
    }
    case legate::mapping::StoreTarget::ZCMEM: {
      ext_alloc =
        legate::ExternalAllocation::create_zcmem(d_alloc, BYTES, false, std::move(deleter));
      break;
    }
    default: {
      CHECK_CUDA(cudaFree(d_alloc));
      return;
    }
  }

  ASSERT_EQ(ext_alloc.size(), BYTES);
  ASSERT_FALSE(ext_alloc.read_only());
  ASSERT_EQ(ext_alloc.ptr(), d_alloc);
  ASSERT_EQ(ext_alloc.target(), store_target);

  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store(legate::Shape{TILE_SIZE}, legate::uint64(), ext_alloc);
  auto p_store = store.get_physical_store();
  auto shape   = p_store.shape<1>();

  if (shape.empty()) {
    return;
  }

  auto acc = p_store.read_write_accessor<std::uint64_t, 1>();

  for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
    // TODO(issue 464)
    // ASSERT_EQ(acc[*it], INIT_VALUE);
    // acc[*it] = INIT_VALUE - 1;
    // ASSERT_EQ(acc[*it], INIT_VALUE - 1);
  }

  store.detach();
#endif
}

}  // namespace

TEST_F(IndexAttach, CPU)
{
  constexpr std::int64_t VAL1 = 42;
  constexpr std::int64_t VAL2 = 84;
  constexpr std::size_t BYTES = TILE_SIZE * sizeof(std::int64_t);

  std::vector<std::int64_t> alloc1(TILE_SIZE, VAL1), alloc2(TILE_SIZE, VAL2);
  auto runtime = legate::Runtime::get_runtime();

  auto ext_alloc1 = legate::ExternalAllocation::create_sysmem(alloc1.data(), BYTES);
  auto ext_alloc2 = legate::ExternalAllocation::create_sysmem(alloc2.data(), BYTES);

  auto [store, _] = runtime->create_store(
    legate::Shape{TILE_SIZE * 2 * runtime->node_count()},
    legate::tuple<std::uint64_t>{TILE_SIZE},
    legate::int64(),
    {{ext_alloc1, legate::tuple<std::uint64_t>{runtime->node_id() * 2}},
     {ext_alloc2, legate::tuple<std::uint64_t>{(runtime->node_id() * 2) + 1}}});

  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<std::int64_t, 1>();
  auto shape   = p_store.shape<1>();

  for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
    ASSERT_EQ(acc[*it], (static_cast<std::size_t>((*it)[0]) / TILE_SIZE) % 2 == 0 ? VAL1 : VAL2);
  }

  store.detach();
}

TEST_F(IndexAttach, GPU)
{
  if (legate::get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    return;
  }

#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  constexpr std::int64_t VAL1 = 42;
  constexpr std::int64_t VAL2 = 84;
  constexpr std::size_t BYTES = TILE_SIZE * sizeof(std::int64_t);

  std::vector<std::int64_t> h_alloc1(TILE_SIZE, VAL1), h_alloc2(TILE_SIZE, VAL2);

  void* d_alloc1 = nullptr;
  void* d_alloc2 = nullptr;

  CHECK_CUDA(cudaMalloc(&d_alloc1, BYTES));
  CHECK_CUDA(cudaMalloc(&d_alloc2, BYTES));

  CHECK_CUDA(cudaMemcpy(d_alloc1, h_alloc1.data(), BYTES, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_alloc2, h_alloc2.data(), BYTES, cudaMemcpyHostToDevice));

  auto deleter = [](void* ptr) noexcept {
    try {
      CHECK_CUDA(cudaFree(ptr));
    } catch (const std::exception& e) {
      LEGATE_ABORT(e.what());
    }
  };
  auto alloc1 =
    legate::ExternalAllocation::create_fbmem(0, d_alloc1, BYTES, true /*read_only*/, deleter);
  auto alloc2 = legate::ExternalAllocation::create_fbmem(
    0, d_alloc2, BYTES, true /*read_only*/, std::move(deleter));
  auto runtime = legate::Runtime::get_runtime();

  auto [store, _] =
    runtime->create_store(legate::Shape{TILE_SIZE * 2 * runtime->node_count()},
                          legate::tuple<std::uint64_t>{TILE_SIZE},
                          legate::int64(),
                          {{alloc1, legate::tuple<std::uint64_t>{runtime->node_id() * 2}},
                           {alloc2, legate::tuple<std::uint64_t>{(runtime->node_id() * 2) + 1}}});

  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<std::int64_t, 1>();
  auto shape   = p_store.shape<1>();

  for (legate::PointInRectIterator<1> it{shape}; it.valid(); ++it) {
    ASSERT_EQ(acc[*it], (static_cast<std::size_t>((*it)[0]) / TILE_SIZE) % 2 == 0 ? VAL1 : VAL2);
  }

  store.detach();
#endif
}

TEST_F(IndexAttach, NegativeAttachSmallerStore)
{
  auto runtime = legate::Runtime::get_runtime();
  std::vector<std::int64_t> buf(TILE_SIZE - 1, 0);
  auto alloc = legate::ExternalAllocation::create_sysmem(
    buf.data(), buf.size() * sizeof(decltype(buf)::value_type));

  // Trying to attach a buffer smaller than what the sub-store requires
  ASSERT_THROW((void)runtime->create_store(legate::Shape{TILE_SIZE},
                                           legate::tuple<std::uint64_t>{TILE_SIZE},
                                           legate::int64(),
                                           {{alloc, legate::tuple<std::uint64_t>{0}}}),
               std::invalid_argument);
}

TEST_F(IndexAttach, NegativeAttachNonexistStore)
{
  auto runtime = legate::Runtime::get_runtime();
  std::vector<std::int64_t> buf(TILE_SIZE, 0);
  auto alloc = legate::ExternalAllocation::create_sysmem(
    buf.data(), buf.size() * sizeof(decltype(buf)::value_type));

  // Trying to attach a buffer to a non-existent sub-store
  ASSERT_THROW((void)runtime->create_store(legate::Shape{TILE_SIZE},
                                           legate::tuple<std::uint64_t>{TILE_SIZE},
                                           legate::int64(),
                                           {{alloc, legate::tuple<std::uint64_t>{1}}}),
               std::out_of_range);
}

TEST_F(IndexAttach, NegativeDuplicateAttach)
{
  auto runtime = legate::Runtime::get_runtime();
  std::vector<std::int64_t> buf(TILE_SIZE, 0);
  auto alloc = legate::ExternalAllocation::create_sysmem(
    buf.data(), buf.size() * sizeof(decltype(buf)::value_type));

  // Trying to attach multiple buffers to the same sub-store
  ASSERT_THROW((void)runtime->create_store(legate::Shape{TILE_SIZE},
                                           legate::tuple<std::uint64_t>{TILE_SIZE},
                                           legate::int64(),
                                           {{alloc, legate::tuple<std::uint64_t>{0}},
                                            {alloc, legate::tuple<std::uint64_t>{0}}}),
               std::invalid_argument);
}

TEST_F(IndexAttach, NegativeAttachVariableStore)
{
  auto runtime = legate::Runtime::get_runtime();

  // Trying to attach buffer with variable size to store
  constexpr const char value[] = "hello world";
  auto ext_alloc               = legate::ExternalAllocation::create_sysmem(value, sizeof(value));

  ASSERT_THROW(
    (void)runtime->create_store(legate::Shape{sizeof(value)}, legate::string_type(), ext_alloc),
    std::invalid_argument);
}

TEST_F(IndexAttach, SysmemAccessByTask)
{
  do_test<std::uint64_t>(10);
  do_test<std::int16_t>(-10);
  do_test<float>(100.0F);
  do_test<double>(10000.8);
  do_test<__half>(static_cast<__half>(0.9F));
  do_test<complex<float>>({15, 20});
  do_test<complex<double>>({-3.9, 5.8});
}

TEST_F(IndexAttach, MutuableSysmemAccessByTask)
{
  constexpr std::size_t BYTES = TILE_SIZE * sizeof(std::uint64_t);
  auto buffer                 = std::make_unique<std::uint64_t[]>(BYTES);
  void* raw_buffer            = static_cast<void*>(buffer.get());
  auto deleter                = [](void* ptr) noexcept {
    auto ext_value_ptr = static_cast<std::uint64_t*>(ptr);

    // changes to the store in task propagated back to the attached allocation
    ASSERT_EQ(*(ext_value_ptr), INIT_VALUE);
  };

  ASSERT_NE(raw_buffer, nullptr);
  std::memset(raw_buffer, 0, BYTES);
  test_sysmem<std::uint64_t>(raw_buffer, 0, BYTES, false, std::move(deleter));
}

TEST_F(IndexAttach, MutableFbmemAccess)
{
  test_gpu_mutuable_access(legate::mapping::StoreTarget::FBMEM);
}

TEST_F(IndexAttach, MutableZcmemAccess)
{
  test_gpu_mutuable_access(legate::mapping::StoreTarget::ZCMEM);
}

TEST_F(IndexAttach, InvalidCreation)
{
  void* ptr = nullptr;

  ASSERT_THROW(static_cast<void>(legate::ExternalAllocation::create_sysmem(ptr, 10)),
               std::invalid_argument);
#if LEGATE_DEFINED(LEGATE_USE_CUDA)
  ASSERT_THROW(static_cast<void>(legate::ExternalAllocation::create_zcmem(ptr, 10)),
               std::invalid_argument);
  if (legate::get_machine().count(legate::mapping::TaskTarget::GPU) > 0) {
    ASSERT_THROW(static_cast<void>(legate::ExternalAllocation::create_fbmem(0, ptr, 10)),
                 std::invalid_argument);
  } else {
    ASSERT_THROW(static_cast<void>(legate::ExternalAllocation::create_fbmem(0, ptr, 10)),
                 std::out_of_range);
  }
#else
  ASSERT_THROW(static_cast<void>(legate::ExternalAllocation::create_zcmem(ptr, 10)),
               std::runtime_error);
  ASSERT_THROW(static_cast<void>(legate::ExternalAllocation::create_fbmem(0, ptr, 10)),
               std::runtime_error);
#endif
}

// NOLINTEND(readability-magic-numbers)

}  // namespace index_attach
