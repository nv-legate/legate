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

#include <legate/data/physical_store.h>
#include <legate/experimental/io/kvikio/detail/tile_by_offsets.h>
#include <legate/type/type_info.h>
#include <legate/utilities/detail/linearize.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/span.h>

#include <kvikio/file_handle.hpp>

#include <cstdint>
#include <string>
#include <sys/types.h>  // off_t

namespace legate::experimental::io::kvikio::detail {

namespace {

/**
 * @brief Functor for tiling read Legate store by offsets into a single file
 *
 * @param context The Legate task context
 * @param store The Legate output store
 */
class TileByOffsetsReadFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(legate::TaskContext context, legate::PhysicalStore* store) const;
};

template <legate::Type::Code CODE, std::int32_t DIM>
void TileByOffsetsReadFn::operator()(legate::TaskContext context,
                                     legate::PhysicalStore* store) const
{
  auto&& shape            = store->shape<DIM>();
  const auto shape_volume = shape.volume();

  if (shape_volume == 0) {
    return;
  }

  using DTYPE = legate::type_of_t<CODE>;

  const auto nbytes = shape_volume * sizeof(DTYPE);
  // We know that the accessor is contiguous because we set `policy.exact = true`
  // in `Mapper::store_mappings()`.
  auto&& task_index    = context.get_task_index();
  auto&& launch_domain = context.get_launch_domain();
  const auto path      = context.scalar(0).value<std::string_view>();
  const auto flatten_task_index =
    legate::detail::linearize(launch_domain.lo(), launch_domain.hi(), task_index);
  const Span<const std::uint64_t> offsets = context.scalar(1).values<std::uint64_t>();

  auto f            = ::kvikio::FileHandle{std::string{path}, "r"};
  auto* data        = store->write_accessor<DTYPE, DIM>().ptr(shape);
  const auto offset = offsets[flatten_task_index];

  if (store->target() == mapping::StoreTarget::FBMEM) {
    static_cast<void>(
      f.read_async(data, nbytes, static_cast<::off_t>(offset), 0, context.get_task_stream()));
  } else {
    f.pread(data, nbytes, offset).wait();
  }
}

}  // namespace

/*static*/ void TileByOffsetsRead::cpu_variant(legate::TaskContext context)
{
  auto store = context.output(0).data();

  legate::double_dispatch(store.dim(), store.code(), TileByOffsetsReadFn{}, context, &store);
}

/*static*/ void TileByOffsetsRead::omp_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
  cpu_variant(context);
}

/*static*/ void TileByOffsetsRead::gpu_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the CPU variant.
  cpu_variant(context);
}

}  // namespace legate::experimental::io::kvikio::detail
