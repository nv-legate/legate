/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/io/hdf5/detail/read.h>

#include <legate/io/hdf5/detail/util.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/detail/types.h>  // for Type::Code formatter
#include <legate/type/type_traits.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <highfive/H5File.hpp>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace legate::io::hdf5::detail {

namespace {

void read_hdf5_file(const HDF5GlobalLock&,
                    const HighFive::DataSet& dataset,
                    const std::vector<std::size_t>& offset,
                    const std::vector<std::size_t>& count,
                    void* dst)
{
  auto&& dspace = dataset.getSpace();
  // Re-use the datatype that we get from the dataset. In this case, HDF5 will simply believe
  // that `dst` points to what we claim it does.
  //
  // We do this because we have already selected the appropriate datatype while launching the
  // task, so we don't want HDF5 to get too fancy on us. For example, when the data is stored
  // as OPAQUE, we try to read it as std::byte. In this case HDF5 would complain because it (by
  // design) does not know how to convert OPAQUE to std::byte, but we know that this is in fact
  // the right type to read with.
  auto&& dtype = dataset.getDataType();

  // Handle scalar & null dataspaces
  if (0 == dspace.getNumberDimensions()) {
    // Scalar
    if (1 == dspace.getElementCount()) {
      dataset.read_raw(dst, dtype);
    }
  } else {
    // Read selected part of the hdf5 file
    dataset.select(offset, count).read_raw(dst, dtype);
  }
}

/**
 * @brief Functor that implements Hdf5ReadTask
 *
 * @tparam DstIsCUDA Whether the operation is reading to host or device memory
 * @param context The Legate task context
 * @param store The Legate store to read into
 */
class HDF5ReadFn {
  // We define `is_supported` to handle unsupported dtype through SFINAE, not sure why
  // clang-tidy complains about this being a C-style cast though...
  template <legate::Type::Code CODE>
  static constexpr bool IS_SUPPORTED =
    !legate::is_complex<CODE>::value;  // NOLINT(google-readability-casting)

 public:
  template <legate::Type::Code CODE,
            std::int32_t DIM,
            std::enable_if_t<IS_SUPPORTED<CODE>>* = nullptr>  // NOLINT(google-readability-casting)
  void operator()(const legate::TaskContext& context,
                  legate::PhysicalStore* store,
                  bool is_device) const
  {
    // TODO(jfaibussowit)
    // Remove this once binary type access is merged
    using DTYPE =
      std::conditional_t<CODE == Type::Code::BINARY, std::byte, legate::type_of_t<CODE>>;

    auto&& shape = store->shape<DIM>();

    if (shape.volume() == 0) {
      return;
    }

    // Find file offset and size of each dimension
    std::vector<std::size_t> offset{};
    std::vector<std::size_t> count{};
    std::size_t total_count = 1;

    offset.reserve(DIM);
    count.reserve(DIM);
    for (std::int32_t i = 0; i < DIM; ++i) {
      offset.emplace_back(shape.lo[i]);
      total_count *= count.emplace_back(shape.hi[i] - shape.lo[i] + 1);
    }

    const auto gds_on = legate::detail::Config::get_config().io_use_vfd_gds();

    const auto filepath     = context.scalar(0).value<std::string_view>();
    const auto dataset_name = context.scalar(1).value<std::string_view>();
    auto* dst               = store
                  ->write_accessor<DTYPE,
                                   DIM,
                                   // TODO(jfaibussowit)
                                   // Remove this once binary type access is merged
                                   /* VALIDATE_TYPE */ CODE != Type::Code::BINARY>()
                  .ptr(shape);

    // Open selected part of the hdf5 file
    const auto f       = open_hdf5_file({}, std::string{filepath}, gds_on);
    const auto dataset = f.getDataSet(std::string{dataset_name});

    if (is_device) {
      if (gds_on) {
        read_hdf5_file({}, dataset, offset, count, dst);
      } else {
        // Otherwise, we read into a bounce buffer
        auto bounce_buffer = create_buffer<DTYPE>(total_count, Memory::Z_COPY_MEM);
        auto* ptr          = bounce_buffer.ptr(0);
        auto stream        = context.get_task_stream();

        read_hdf5_file({}, dataset, offset, count, ptr);
        // And then copy from the bounce buffer to the GPU
        legate::detail::Runtime::get_runtime()->get_cuda_driver_api()->mem_cpy_async(
          dst, ptr, shape.volume() * sizeof(DTYPE), stream);
      }
    } else {
      // When running on a CPU, we read directly into the destination memory
      read_hdf5_file({}, dataset, offset, count, dst);
    }
  }

  template <legate::Type::Code CODE,
            std::int32_t DIM,
            std::enable_if_t<!IS_SUPPORTED<CODE>>* = nullptr>
  void operator()(const legate::TaskContext&, legate::PhysicalStore*, bool)
  {
    throw legate::detail::TracedException<std::runtime_error>{
      fmt::format("HDF5 read not supported for {}", CODE)};
  }

 private:
};

// TODO(jfaibussowit)
// Remove this and merge https://github.com/nv-legate/legate.internal/pull/1604
template <std::int32_t DIM>
class TypeDispatcher : public legate::detail::InnerTypeDispatchFn<DIM> {
 public:
  using base = legate::detail::InnerTypeDispatchFn<DIM>;

  template <typename Functor, typename... Fnargs>
  constexpr decltype(auto) operator()(legate::Type::Code code, Functor&& f, Fnargs&&... args)
  {
    if (code == legate::Type::Code::BINARY) {
      return f.template operator()<legate::Type::Code::BINARY, DIM>(std::forward<Fnargs>(args)...);
    }
    return static_cast<base&>(*this)(code, std::forward<Functor>(f), std::forward<Fnargs>(args)...);
  }
};

void task_body(const legate::TaskContext& context, bool is_device)
{
  auto store = context.output(0).data();

  // The below is just an unrolled
  //
  // legate::double_dispatch(store.dim(), store.code(), HDF5ReadFn{}, context, &store,
  // is_device);
  //
  // because  double_dispatch() does not support Type::Code::BINARY yet, and it won't until
  // https://github.com/nv-legate/legate.internal/pull/1604 is resolved/merged. This cludge was
  // done for the 25.01 release.

#define TYPE_DISPATCH(__dim__) \
  case __dim__:                \
    return TypeDispatcher<__dim__>{}(store.code(), HDF5ReadFn{}, context, &store, is_device);

  switch (const auto dim = store.dim()) {
    LEGION_FOREACH_N(TYPE_DISPATCH);
    default: {  // legate-lint: no-switch-default
      legate::detail::throw_unsupported_dim(dim);
    }
  }
}

}  // namespace

/*static*/ void HDF5Read::cpu_variant(legate::TaskContext context) { task_body(context, false); }

/*static*/ void HDF5Read::omp_variant(legate::TaskContext context) { cpu_variant(context); }

/*static*/ void HDF5Read::gpu_variant(legate::TaskContext context) { task_body(context, true); }

}  // namespace legate::io::hdf5::detail
