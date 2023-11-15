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
#include "legateio.h"
#include "util.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace legateio {

namespace {

struct read_fn {
  template <legate::Type::Code CODE, int32_t DIM>
  void operator()(legate::Store& output, const fs::path& path)
  {
    using VAL = legate::legate_type_of<CODE>;

    legate::Rect<DIM> shape = output.shape<DIM>();

    if (shape.empty()) {
      return;
    }

    std::ifstream in(path, std::ios::binary | std::ios::in);

    legate::Point<DIM> extents;
    for (int32_t idx = 0; idx < DIM; ++idx) {
      in.read(reinterpret_cast<char*>(&extents[idx]), sizeof(legate::coord_t));
    }

    // Since the shape is already fixed on the Python side, the sub-store's extents should be the
    // same as what's stored in the file
    assert(shape.hi - shape.lo + legate::Point<DIM>::ONES() == extents);

    logger.print() << "Read a sub-array of rect " << shape << " from " << path;

    auto acc = output.write_accessor<VAL, DIM>();
    for (legate::PointInRectIterator<DIM> it(shape, false /*fortran_order*/); it.valid(); ++it) {
      auto ptr = acc.ptr(*it);
      in.read(reinterpret_cast<char*>(ptr), sizeof(VAL));
    }
  }
};

}  // namespace

class ReadEvenTilesTask : public Task<ReadEvenTilesTask, READ_EVEN_TILES> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto dirname = context.scalar(0).value<std::string>();
    auto output  = context.output(0).data();

    auto path = get_unique_path_for_task_index(context, output.dim(), dirname);
    // double_dispatch converts the first two arguments to non-type template arguments
    legate::double_dispatch(output.dim(), output.code(), read_fn{}, output, path);
  }
};

}  // namespace legateio

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legateio::ReadEvenTilesTask::register_variants();
}

}  // namespace
