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

#include <legate/experimental/io/kvikio/detail/tile.h>

#include <legate/type/type_traits.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/span.h>

#include <kvikio/file_handle.hpp>

#include <fmt/ranges.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string_view>
#include <utility>

namespace legate::experimental::io::kvikio::detail {

namespace {

class TileReadWriteFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(legate::TaskContext context, bool is_read_op, legate::PhysicalStore* store) const;

 private:
  [[nodiscard]] static DomainPoint get_tile_coord_(const DomainPoint& task_index,
                                                   Span<const std::uint64_t> tile_start);
  [[nodiscard]] static std::filesystem::path get_file_path_(std::string_view dirpath,
                                                            const DomainPoint& tile_coord);
};

DomainPoint TileReadWriteFn::get_tile_coord_(const DomainPoint& task_index,
                                             Span<const std::uint64_t> tile_start)
{
  auto ret = task_index;

  for (int i = 0; i < task_index.dim; ++i) {
    ret[i] += static_cast<coord_t>(tile_start[i]);
  }
  return ret;
}

std::filesystem::path TileReadWriteFn::get_file_path_(std::string_view dirpath,
                                                      const DomainPoint& tile_coord)
{
  // dirpath / coord0.coord1.coord2.coord3
  return fmt::format("{}{}{}",
                     dirpath,
                     std::filesystem::path::preferred_separator,
                     fmt::join(tile_coord.point_data, tile_coord.point_data + tile_coord.dim, "."));
}

// ==========================================================================================

template <legate::Type::Code CODE, std::int32_t DIM>
void TileReadWriteFn::operator()(legate::TaskContext context,
                                 bool is_read_op,
                                 legate::PhysicalStore* store) const
{
  auto&& shape            = store->shape<DIM>();
  const auto shape_volume = shape.volume();

  if (shape_volume == 0) {
    return;
  }

  using DTYPE = legate::type_of_t<CODE>;

  auto&& task_index     = context.get_task_index();
  auto path             = context.scalar(0).value<std::string_view>();
  const auto tile_start = context.scalar(1).values<std::uint64_t>();
  const auto tile_coord = get_tile_coord_(task_index, tile_start);
  const auto filepath   = get_file_path_(path, tile_coord);
  auto f                = ::kvikio::FileHandle{filepath, is_read_op ? "r" : "w"};
  const auto nbytes     = shape_volume * sizeof(DTYPE);
  // We know that the accessor is contiguous because we set `policy.exact = true`
  // in `mapper.cc`.
  if (is_read_op) {
    auto* data = store->write_accessor<DTYPE, DIM>().ptr(shape);

    if (store->target() == mapping::StoreTarget::FBMEM) {
      static_cast<void>(f.read_async(data, nbytes, 0, 0, context.get_task_stream()));
    } else {
      f.pread(data, nbytes).wait();
    }
  } else {
    const auto* data = store->read_accessor<DTYPE, DIM>().ptr(shape);

    if (store->target() == mapping::StoreTarget::FBMEM) {
      static_cast<void>(
        f.write_async(const_cast<DTYPE*>(data), nbytes, 0, 0, context.get_task_stream()));
    } else {
      f.pwrite(data, nbytes).wait();
    }
  }
}

}  // namespace

// ==========================================================================================

/*static*/ void TileRead::cpu_variant(legate::TaskContext context)
{
  auto store = context.output(0).data();

  legate::double_dispatch(
    store.dim(), store.code(), TileReadWriteFn{}, context, /*read*/ true, &store);
}

/*static*/ void TileRead::omp_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
  cpu_variant(context);
}

/*static*/ void TileRead::gpu_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
  cpu_variant(context);
}

// ==========================================================================================

/*static*/ void TileWrite::cpu_variant(legate::TaskContext context)
{
  auto store = context.input(0).data();

  legate::double_dispatch(
    store.dim(), store.code(), TileReadWriteFn{}, context, /*read*/ false, &store);
}

/*static*/ void TileWrite::omp_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
  cpu_variant(context);
}

/*static*/ void TileWrite::gpu_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
  cpu_variant(context);
}

}  // namespace legate::experimental::io::kvikio::detail
