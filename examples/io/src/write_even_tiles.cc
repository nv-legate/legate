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

#include "core/utilities/span.h"

#include "legate_library.h"
#include "legateio.h"
#include "util.h"

#include <filesystem>
#include <fstream>

namespace fs = std::filesystem;

namespace legateio {

namespace {

void write_header(std::ofstream& out,
                  legate::Type::Code type_code,
                  const legate::Span<const int32_t>& shape,
                  const legate::Span<const int32_t>& tile_shape)
{
  assert(shape.size() == tile_shape.size());
  int32_t dim = shape.size();
  // Dump the type code, the array's shape and the tile shape to the header
  out.write(reinterpret_cast<const char*>(&type_code), sizeof(int32_t));
  out.write(reinterpret_cast<const char*>(&dim), sizeof(int32_t));
  for (auto& v : shape) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(int32_t));
  }
  for (auto& v : tile_shape) {
    out.write(reinterpret_cast<const char*>(&v), sizeof(int32_t));
  }
}

}  // namespace

class WriteEvenTilesTask : public Task<WriteEvenTilesTask, WRITE_EVEN_TILES> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto dirname                           = context.scalar(0).value<std::string>();
    legate::Span<const int32_t> shape      = context.scalar(1).values<int32_t>();
    legate::Span<const int32_t> tile_shape = context.scalar(2).values<int32_t>();
    auto input                             = context.input(0).data();

    auto launch_domain = context.get_launch_domain();
    auto task_index    = context.get_task_index();
    auto is_first_task = context.is_single_task() || task_index == launch_domain.lo();

    if (is_first_task) {
      auto header = fs::path(dirname) / ".header";
      logger.print() << "Write to " << header;
      std::ofstream out(header, std::ios::binary | std::ios::out | std::ios::trunc);
      write_header(out, input.code(), shape, tile_shape);
    }

    write_to_file(context, dirname, input);
  }
};

}  // namespace legateio

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legateio::WriteEvenTilesTask::register_variants();
}

}  // namespace
