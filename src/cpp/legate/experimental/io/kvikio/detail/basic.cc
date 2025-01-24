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

#include <legate/experimental/io/kvikio/detail/basic.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/dispatch.h>

#include <kvikio/file_handle.hpp>

#include <string>
#include <string_view>
#include <sys/types.h>  // off_t
#include <type_traits>

namespace legate::experimental::io::kvikio::detail {

namespace {

class KvikioReadWriteFn {
 public:
  template <legate::Type::Code CODE>
  void operator()(const legate::TaskContext& context,
                  std::string_view path,
                  legate::PhysicalStore* store,
                  bool is_read_op) const;
};

template <legate::Type::Code CODE>
void KvikioReadWriteFn::operator()(const legate::TaskContext& context,
                                   std::string_view path,
                                   legate::PhysicalStore* store,
                                   bool is_read_op) const
{
  LEGATE_ASSERT(store->dim() == 1);
  const auto shape  = store->shape<1>();
  const auto volume = shape.volume();
  // No need to do anything if we're responsible for an empty sub-range.
  if (volume == 0) {
    return;
  }

  using DTYPE = legate::type_of_t<CODE>;

  const auto nbytes = volume * sizeof(DTYPE);
  const auto offset = static_cast<std::size_t>(shape.lo) * sizeof(DTYPE);
  static_assert(
    !std::is_constructible_v<::kvikio::FileHandle, std::string_view, const std::string&>,
    "can use std::string_view as filepath argument instead of std::string");
  auto f = ::kvikio::FileHandle{std::string{path}, is_read_op ? "r" : "r+"};

  // We know that the accessor is contiguous because we set `policy.exact = true`
  // in `Mapper::store_mappings()`.
  if (is_read_op) {
    auto* data = store->write_accessor<DTYPE, 1>().ptr(shape);

    if (store->target() == mapping::StoreTarget::FBMEM) {
      static_cast<void>(
        f.read_async(data, nbytes, static_cast<::off_t>(offset), 0, context.get_task_stream()));
    } else {
      f.pread(data, nbytes, offset).wait();
    }
  } else {
    const auto* data = store->read_accessor<DTYPE, 1>().ptr(shape);

    if (store->target() == mapping::StoreTarget::FBMEM) {
      static_cast<void>(f.write_async(const_cast<DTYPE*>(data),
                                      nbytes,
                                      static_cast<::off_t>(offset),
                                      0,
                                      context.get_task_stream()));
    } else {
      f.pwrite(data, nbytes, offset).wait();
    }
  }
}

}  // namespace

/*static*/ void BasicRead::cpu_variant(legate::TaskContext context)
{
  const auto path = context.scalar(0).value<std::string_view>();
  auto store      = context.output(0).data();

  legate::type_dispatch(
    store.code(), KvikioReadWriteFn{}, context, path, &store, /* read_op */ true);
}

/*static*/ void BasicRead::omp_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the
  // CPU variant.
  cpu_variant(context);
}

/*static*/ void BasicRead::gpu_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the
  // CPU variant.
  cpu_variant(context);
}

// ==========================================================================================

/*static*/ void BasicWrite::cpu_variant(legate::TaskContext context)
{
  const auto path = context.scalar(0).value<std::string_view>();
  auto store      = context.input(0).data();

  legate::type_dispatch(
    store.code(), KvikioReadWriteFn{}, context, path, &store, /* read_op */ false);
}

/*static*/ void BasicWrite::omp_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the
  // CPU variant.
  cpu_variant(context);
}

/*static*/ void BasicWrite::gpu_variant(legate::TaskContext context)
{
  // Since KvikIO supports both GPU and CPU memory seamlessly, we reuse the
  // CPU variant.
  cpu_variant(context);
}

}  // namespace legate::experimental::io::kvikio::detail
