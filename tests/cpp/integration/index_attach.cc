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
#include "utilities/utilities.h"

#include <gtest/gtest.h>

#if LegateDefined(LEGATE_USE_CUDA)
#include "core/cuda/cuda_help.h"

#include <cuda_runtime.h>
#endif

#include <vector>

namespace index_attach {

using IndexAttach = DefaultFixture;

constexpr size_t TILE_SIZE = 5;

TEST_F(IndexAttach, CPU)
{
  constexpr int64_t VAL1 = 42;
  constexpr int64_t VAL2 = 84;
  constexpr size_t BYTES = TILE_SIZE * sizeof(int64_t);

  std::vector<int64_t> alloc1(TILE_SIZE, VAL1), alloc2(TILE_SIZE, VAL2);
  auto runtime = legate::Runtime::get_runtime();

  auto ext_alloc1 = legate::ExternalAllocation::create_sysmem(alloc1.data(), BYTES);
  auto ext_alloc2 = legate::ExternalAllocation::create_sysmem(alloc2.data(), BYTES);

  auto [store, _] =
    runtime->create_store(legate::Shape{TILE_SIZE * 2},
                          legate::Shape{TILE_SIZE},
                          legate::int64(),
                          {{ext_alloc1, legate::Shape{0}}, {ext_alloc2, legate::Shape{1}}});

  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<int64_t, 1>();
  auto shape   = p_store.shape<1>();
  for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
    EXPECT_EQ(acc[*it], static_cast<size_t>((*it)[0]) < TILE_SIZE ? VAL1 : VAL2);
  }

  store.detach();
}

TEST_F(IndexAttach, GPU)
{
  if (legate::get_machine().count(legate::mapping::TaskTarget::GPU) == 0) {
    return;
  }

#if LegateDefined(LEGATE_USE_CUDA)
  constexpr int64_t VAL1 = 42;
  constexpr int64_t VAL2 = 84;
  constexpr size_t BYTES = TILE_SIZE * sizeof(int64_t);

  std::vector<int64_t> h_alloc1(TILE_SIZE, VAL1), h_alloc2(TILE_SIZE, VAL2);

  void* d_alloc1 = nullptr;
  void* d_alloc2 = nullptr;

  CHECK_CUDA(cudaMalloc(&d_alloc1, BYTES));
  CHECK_CUDA(cudaMalloc(&d_alloc2, BYTES));

  CHECK_CUDA(cudaMemcpy(d_alloc1, h_alloc1.data(), BYTES, cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(d_alloc2, h_alloc2.data(), BYTES, cudaMemcpyHostToDevice));

  auto deleter = [](void* ptr) noexcept { CHECK_CUDA(cudaFree(ptr)); };
  auto alloc1 =
    legate::ExternalAllocation::create_fbmem(0, d_alloc1, BYTES, true /*read_only*/, deleter);
  auto alloc2 =
    legate::ExternalAllocation::create_fbmem(0, d_alloc2, BYTES, true /*read_only*/, deleter);
  auto runtime = legate::Runtime::get_runtime();

  auto [store, _] = runtime->create_store(legate::Shape{TILE_SIZE * 2},
                                          legate::Shape{TILE_SIZE},
                                          legate::int64(),
                                          {{alloc1, legate::Shape{0}}, {alloc2, legate::Shape{1}}});

  auto p_store = store.get_physical_store();
  auto acc     = p_store.read_accessor<int64_t, 1>();
  auto shape   = p_store.shape<1>();
  for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
    EXPECT_EQ(acc[*it], static_cast<size_t>((*it)[0]) < TILE_SIZE ? VAL1 : VAL2);
  }

  store.detach();
#endif
}

}  // namespace index_attach
