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

#include <fstream>

#include "legateio.h"
#include "util.h"

#include "core/type/type_traits.h"
#include "core/utilities/dispatch.h"

namespace fs = std::filesystem;

namespace legateio {

namespace {

struct write_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(const legate::Store& store, const fs::path& path)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape = store.shape<DIM>();
    auto empty = shape.empty();
    auto extents =
      empty ? legate::Point<DIM>::ZEROES() : shape.hi - shape.lo + legate::Point<DIM>::ONES();

    int32_t dim  = DIM;
    int32_t code = store.code<int32_t>();

    logger.print() << "Write a sub-array " << shape << " to " << path;

    std::ofstream out(path, std::ios::binary | std::ios::out | std::ios::trunc);
    // Each file for a chunk starts with the extents
    for (int32_t idx = 0; idx < DIM; ++idx)
      out.write(reinterpret_cast<const char*>(&extents[idx]), sizeof(legate::coord_t));

    if (empty) return;
    auto acc = store.read_accessor<VAL, DIM>();
    // The iteration order here should be consistent with that in the reader task, otherwise
    // the read data can be transposed.
    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto ptr = acc.ptr(*it);
      out.write(reinterpret_cast<const char*>(ptr), sizeof(VAL));
    }
  }
};

}  // namespace

std::filesystem::path get_unique_path_for_task_index(const legate::TaskContext& context,
                                                     int32_t ndim,
                                                     const std::string& dirname)
{
  auto task_index = context.get_task_index();
  // If this was a single task, we use (0, ..., 0) for the task index
  if (context.is_single_task()) {
    task_index     = legate::DomainPoint();
    task_index.dim = ndim;
  }

  std::stringstream ss;
  for (int32_t idx = 0; idx < task_index.dim; ++idx) {
    if (idx != 0) ss << ".";
    ss << task_index[idx];
  }
  auto filename = ss.str();

  return fs::path(dirname) / filename;
}

void write_to_file(legate::TaskContext& task_context,
                   const std::string& dirname,
                   const legate::Store& store)
{
  auto path = get_unique_path_for_task_index(task_context, store.dim(), dirname);
  // double_dispatch converts the first two arguments to non-type template arguments
  legate::double_dispatch(store.dim(), store.code(), write_fn{}, store, path);
}

}  // namespace legateio
