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

#include "core/utilities/dispatch.h"

#include "legate_library.h"
#include "legateio.h"

#include <fstream>

namespace legateio {

namespace {

struct write_fn {
  template <legate::Type::Code CODE>
  void operator()(const legate::Store& input, const std::string& filename)
  {
    using VAL = legate::legate_type_of<CODE>;

    auto shape  = input.shape<1>();
    auto code   = input.code<int64_t>();
    size_t size = shape.volume();

    // Store the type code and the number of elements in the array at the beginning of the file
    std::ofstream out(filename, std::ios::binary | std::ios::out | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(&code), sizeof(int64_t));
    out.write(reinterpret_cast<const char*>(&size), sizeof(size_t));

    auto acc = input.read_accessor<VAL, 1>();
    for (legate::PointInRectIterator<1> it(shape); it.valid(); ++it) {
      auto ptr = acc.ptr(*it);
      out.write(reinterpret_cast<const char*>(ptr), sizeof(VAL));
    }
  }
};

}  // namespace

class WriteFileTask : public Task<WriteFileTask, WRITE_FILE> {
 public:
  static void cpu_variant(legate::TaskContext context)
  {
    auto filename = context.scalar(0).value<std::string>();
    auto input    = context.input(0).data();
    logger.print() << "Write to " << filename;

    legate::type_dispatch(input.code(), write_fn{}, input, filename);
  }
};

}  // namespace legateio

namespace {

static void __attribute__((constructor)) register_tasks()
{
  legateio::WriteFileTask::register_variants();
}

}  // namespace
