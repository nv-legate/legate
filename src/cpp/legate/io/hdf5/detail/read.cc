/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/read.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/detail/types.h>  // for Type::Code formatter
#include <legate/type/type_traits.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace legate::io::hdf5::detail {

namespace {

void read_hdf5_file(legate::detail::ZStringView filepath,
                    bool gds_on,
                    legate::detail::ZStringView dataset_name,
                    Span<const hsize_t> offset,
                    Span<const hsize_t> count,
                    void* dst)
{
  const auto f = [&] {
    if (gds_on) {
      constexpr auto BLOCK_SIZE = 4096;
      constexpr auto BUF_SIZE   = 16 * 1024 * 1024;
      auto props                = wrapper::HDF5FileAccessPropertyList{};

      props.set_gds(BLOCK_SIZE, BLOCK_SIZE, BUF_SIZE);
      return wrapper::HDF5File{filepath, wrapper::HDF5File::OpenMode::READ_ONLY, props};
    }
    return wrapper::HDF5File{filepath, wrapper::HDF5File::OpenMode::READ_ONLY};
  }();
  // Open selected part of the hdf5 file
  const auto dataset = f.data_set(dataset_name);
  auto dspace        = dataset.data_space();

  // Handle scalar & null dataspaces
  if (dspace.extents().empty()) {
    // Scalar
    if (1 == dspace.element_count()) {
      dataset.read(LEGATE_PURE_H5_ENUM(H5S_ALL),
                   LEGATE_PURE_H5_ENUM(H5S_ALL),
                   LEGATE_PURE_H5_ENUM(H5P_DEFAULT),
                   dst);
    }
    return;
  }

  dspace.select_hyperslab(wrapper::HDF5DataSpace::SelectMode::SELECT_SET, offset, count);

  const auto memspace = wrapper::HDF5DataSpace{count};

  dataset.read(memspace.hid(), dspace.hid(), LEGATE_PURE_H5_ENUM(H5P_DEFAULT), dst);
}

/**
 * @brief Functor that implements Hdf5ReadTask
 *
 * @tparam DstIsCUDA Whether the operation is reading to host or device memory
 * @param context The Legate task context
 * @param store The Legate store to read into
 */
class HDF5ReadFn {
  // We define `is_supported` to handle unsupported dtype through SFINAE. It used to be a
  // static constexpr bool variable, but then NVC++ complained that it was "never used". Making
  // it a function hopefully silences this useless warning.
  template <legate::Type::Code CODE>
  static constexpr bool is_supported_()
  {
    return !legate::is_complex<CODE>::value;
  }

 public:
  template <
    legate::Type::Code CODE,
    std::int32_t DIM,
    typename = std::enable_if_t<is_supported_<CODE>()>>  // NOLINT(google-readability-casting)
  void operator()(const legate::TaskContext& context,
                  legate::PhysicalStore* store,
                  bool is_device) const
  {
    // TODO(jfaibussowit)
    // Remove this once binary type access is merged
    constexpr auto BINARY_TYPE = CODE == Type::Code::BINARY;
    using DTYPE = std::conditional_t<BINARY_TYPE, std::byte, legate::type_of_t<CODE>>;

    auto&& shape = store->shape<DIM>();

    if (shape.volume() == 0) {
      return;
    }

    // Find file offset and size of each dimension
    std::array<hsize_t, DIM> offset{};
    std::array<hsize_t, DIM> count{};

    for (std::uint32_t i = 0; i < DIM; ++i) {
      offset[i] = static_cast<hsize_t>(shape.lo[i]);
      count[i]  = static_cast<hsize_t>(shape.hi[i] - shape.lo[i] + 1);
    }

    const auto gds_on       = legate::detail::Runtime::get_runtime().config().io_use_vfd_gds();
    const auto filepath     = context.scalar(0).value<std::string>();
    const auto dataset_name = context.scalar(1).value<std::string>();
    const auto acc          = store->write_accessor<DTYPE,
                                                    DIM,
                                                    // TODO(jfaibussowit)
                                                    // Remove this once binary type access is merged
                                                    /* VALIDATE_TYPE */ !BINARY_TYPE>();
    // The binary type will default to a field size of 1, but our actual type might be
    // arbitrarily sized. So we need to tell Legion the true size, otherwise we get a bunch of:
    //
    // ERROR: Illegal request for pointer of non-dense rectangle
    // Assertion failed: (false), function ptr, file legion.inl, line 4125.
    const auto type_size = BINARY_TYPE ? store->type().size() : sizeof(DTYPE);
    auto* dst            = acc.ptr(shape, /* field_size */ type_size);

    if ((is_device && gds_on) || !is_device) {
      // When running on a CPU, or using GPU with GDS, we can read directly into the
      // destination memory
      read_hdf5_file(filepath, gds_on, dataset_name, offset, count, dst);
      return;
    }

    // Otherwise, we need to read into a bounce buffer
    const auto total_count = shape.volume();
    auto bounce_buffer     = create_buffer<DTYPE>(total_count, Memory::Z_COPY_MEM);
    auto* const ptr        = bounce_buffer.ptr(0);
    auto stream            = context.get_task_stream();

    read_hdf5_file(filepath, gds_on, dataset_name, offset, count, ptr);
    // And then copy from the bounce buffer to the GPU
    cuda::detail::get_cuda_driver_api()->mem_cpy_async(dst, ptr, total_count * type_size, stream);
  }

  template <legate::Type::Code CODE,
            std::int32_t DIM,
            typename = std::enable_if_t<!is_supported_<CODE>()>>
  void operator()(const legate::TaskContext&, legate::PhysicalStore*, bool)
  {
    throw legate::detail::TracedException<std::runtime_error>{
      fmt::format("HDF5 read not supported for {}", CODE)};
  }
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
