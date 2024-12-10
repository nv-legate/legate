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

#include <legate/io/hdf5/detail/read.h>
#include <legate/io/hdf5/detail/util.h>
#include <legate/mapping/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/detail/type_info.h>  // for Type::Code formatter
#include <legate/type/type_traits.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/env.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <highfive/H5File.hpp>
#include <highfive/H5PropertyList.hpp>

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace legate::io::hdf5::detail {

namespace {

template <typename T>
void read_hdf5_file(const HDF5GlobalLock&,
                    const HighFive::DataSet& dataset,
                    T* dst,
                    const std::vector<std::size_t>& offset,
                    const std::vector<std::size_t>& count)
{
  auto&& dspace = dataset.getSpace();

  // Handle scalar & null dataspaces
  if (0 == dspace.getNumberDimensions()) {
    // Scalar
    if (1 == dspace.getElementCount()) {
      dataset.read_raw(dst);
    }
  } else {
    // Read selected part of the hdf5 file
    auto&& src = dataset.select(offset, count);

    src.read_raw(dst);
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
  void operator()(const legate::TaskContext& context, legate::PhysicalStore* store, bool is_device)
  {
    using DTYPE = legate::type_of_t<CODE>;

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

    const auto gds_on = legate::detail::LEGATE_IO_USE_VFD_GDS.get().value_or(false);

    auto&& filepath     = context.scalar(0).value<std::string>();
    auto&& dataset_name = context.scalar(1).value<std::string>();
    auto* dst           = store->write_accessor<DTYPE, DIM>().ptr(shape);

    // Open selected part of the hdf5 file
    const auto f       = open_hdf5_file({}, filepath, gds_on);
    const auto dataset = f.getDataSet(dataset_name);

    if (is_device) {
      if (gds_on) {
        read_hdf5_file({}, dataset, dst, offset, count);
      } else {
        // Otherwise, we read into a bounce buffer
        auto bounce_buffer = create_buffer<DTYPE>(total_count, Memory::Z_COPY_MEM);
        auto stream        = context.get_task_stream();

        read_hdf5_file({}, dataset, bounce_buffer.ptr(0), offset, count);
        // And then copy from the bounce buffer to the GPU
        legate::detail::Runtime::get_runtime()->get_cuda_driver_api()->mem_cpy_async(
          dst, bounce_buffer.ptr(0), shape.volume() * sizeof(DTYPE), stream);
      }
    } else {
      // When running on a CPU, we read directly into the destination memory
      read_hdf5_file({}, dataset, dst, offset, count);
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

void task_body(const legate::TaskContext& context, bool is_device)
{
  auto store = context.output(0).data();

  legate::double_dispatch(store.dim(), store.code(), HDF5ReadFn{}, context, &store, is_device);
}

}  // namespace

/*static*/ void HDF5Read::cpu_variant(legate::TaskContext context) { task_body(context, false); }

/*static*/ void HDF5Read::omp_variant(legate::TaskContext context) { cpu_variant(context); }

/*static*/ void HDF5Read::gpu_variant(legate::TaskContext context) { task_body(context, true); }

}  // namespace legate::io::hdf5::detail
