/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/write_vds.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <H5Ppublic.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

class FlatDomainPoint {
 public:
  const legate::DomainPoint& dp;
};

}  // namespace

namespace fmt {

template <>
struct formatter<FlatDomainPoint> : public formatter<string_view> {
  format_context::iterator format(const FlatDomainPoint& value, format_context& ctx) const
  {
    const auto& point = value.dp;

    return format_to(
      ctx.out(), "{}", fmt::join(point.point_data, point.point_data + point.dim, "_"));
  }
};

}  // namespace fmt

namespace legate::io::hdf5::detail {

namespace {

/**
 * @brief Actually create the HDF5 file on disk.
 *
 * @param filepath The filename to write.
 * @param extents The extents of the array to write.
 * @param dataset_name The name of the dataset to write.
 * @param type The type of the data.
 * @param gds_on Whether to enable GDS when opening the file.
 * @param ptr A pointer to the beginning of the buffer to write. It must be of size
 * `extents.volume()`.
 */
void write_hdf5_file(const std::string& filepath,
                     Span<const hsize_t> file_space_extents,
                     Span<const hsize_t> mem_space_extents,
                     const std::string& dataset_name,
                     const Type& type,
                     bool gds_on,
                     const void* ptr)
{
  const auto file = [&] {
    if (gds_on) {
      constexpr auto BLOCK_SIZE = 4096;
      constexpr auto BUF_SIZE   = 16 * 1024 * 1024;
      auto props                = wrapper::HDF5FileAccessPropertyList{};

      props.set_gds(BLOCK_SIZE, BLOCK_SIZE, BUF_SIZE);
      return wrapper::HDF5File{filepath, wrapper::HDF5File::OpenMode::OVERWRITE, props};
    }
    return wrapper::HDF5File{filepath, wrapper::HDF5File::OpenMode::OVERWRITE};
  }();
  const auto file_space = wrapper::HDF5DataSpace{file_space_extents};
  const auto dset       = wrapper::HDF5DataSet{file, dataset_name, type, file_space};
  auto mem_space        = wrapper::HDF5DataSpace{mem_space_extents};
  const auto offsets    = legate::detail::SmallVector<hsize_t, LEGATE_MAX_DIM>{
    legate::detail::tags::size_tag, file_space_extents.size(), 0};

  mem_space.select_hyperslab(
    wrapper::HDF5DataSpace::SelectMode::SELECT_SET, offsets, file_space_extents);
  dset.write(mem_space.hid(), file_space.hid(), LEGATE_PURE_H5_ENUM(H5P_DEFAULT), ptr);
}

class HDF5WriteFn {
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
    std::enable_if_t<is_supported_<CODE>()>* = nullptr>  // NOLINT(google-readability-casting)
  void operator()(const legate::TaskContext& context,
                  const legate::PhysicalStore& store,
                  const std::filesystem::path& filepath,
                  const std::string& dataset_name,
                  bool is_device) const
  {
    constexpr auto BINARY_TYPE = CODE == Type::Code::BINARY;
    using T                    = std::conditional_t<BINARY_TYPE, std::byte, type_of_t<CODE>>;
    const auto type            = store.type();
    // If we are copying binary data, then sizeof(*tmp_ptr) will give us sizeof(std::byte),
    // but that's not correct since the underlying binary data might be arbitrarily
    // sized. So we need to use the type size.
    //
    // If not using binary type, type.size() and sizeof(*tmp_ptr) should be equivalent, but
    // we use sizeof() as it's faster.
    const auto type_size = BINARY_TYPE ? type.size() : sizeof(T);
    const auto acc       = store.span_read_accessor<T,
                                                    DIM,
                                                    // TODO(jfaibussowit)
                                                    // Remove this once binary type access is merged
                                                    /* VALIDATE_TYPE */ !BINARY_TYPE>(type_size);

    const auto* const ptr = acc.data_handle();
    const auto gds_on     = legate::detail::Runtime::get_runtime().config().io_use_vfd_gds();
    const auto file_space_extents = [&] {
      auto ret = std::array<hsize_t, DIM>{};

      for (std::size_t i = 0; i < DIM; ++i) {
        ret[i] = static_cast<hsize_t>(acc.extent(i));
      }
      return ret;
    }();

    if ((is_device && gds_on) || !is_device) {
      // HDF5 requires two sizes when defining a memory space:
      // 1. The total extent of the memory region (including unused elements).
      // 2. The extent of the active selection (what maps to the file).
      //
      // The selection size matches `file_space_extents`, but the total memory region may be
      // larger. Visually, the selection is the x's, while the memory region includes both x's
      // and o's:
      //
      //     memspace.       ->  file space
      // [ x x x x o o o o ].    [ x x x x ]
      // [ x x x x o o o o ].    [ x x x x ]
      // [ x x x x o o o o ].    [ x x x x ]
      // [ x x x x o o o o ].    [ x x x x ]
      // [ x x x x o o o o ].    [ x x x x ]
      //
      // Since an mdspan may not be densely packed, o's can appear. The code below uses the
      // strides to compute the full memory shape.
      const auto mem_space_extents = [&] {
        auto ret = std::array<hsize_t, DIM>{};

        ret[0] = static_cast<hsize_t>(acc.extent(0));
        for (std::size_t i = 0; i < DIM - 1; ++i) {
          using index_type = typename std::decay_t<decltype(acc)>::index_type;
          // Strides may be zero if the size of the mdspan is 0, which could cause
          // divide-by-zero.
          ret[i + 1] =
            static_cast<hsize_t>(acc.stride(i) / std::max(acc.stride(i + 1), index_type{1}));
        }
        return ret;
      }();

      write_hdf5_file(
        filepath, file_space_extents, mem_space_extents, dataset_name, type, gds_on, ptr);
      return;
    }

    // ...otherwise we need to copy to a temporary buffer on the host first
    const auto size     = acc.size();
    auto tmp            = create_buffer<std::byte>(size * type_size, Memory::Z_COPY_MEM);
    auto* const tmp_ptr = tmp.ptr(0);
    auto stream         = context.get_task_stream();
    auto&& api          = cuda::detail::get_cuda_driver_api();
    // These extents are a lie (the temp buffer is decidedly 1D), but the size is the same, so
    // HDF5 will treat it as 1-dimensional anyways.
    const auto mem_space_extents = file_space_extents;

    api->mem_cpy_async(tmp_ptr, ptr, size * type_size, stream);
    // Need to synchronize here before we pass to HDF5
    api->stream_synchronize(stream);
    write_hdf5_file(
      filepath, file_space_extents, mem_space_extents, dataset_name, type, gds_on, tmp_ptr);
  }

  template <legate::Type::Code CODE,
            std::int32_t DIM,
            std::enable_if_t<!is_supported_<CODE>()>* = nullptr>
  void operator()(const legate::TaskContext&,
                  const legate::PhysicalStore&,
                  const std::filesystem::path&,
                  const std::string&,
                  bool) const
  {
    throw legate::detail::TracedException<std::runtime_error>{
      fmt::format("HDF5 write not supported for {}", CODE)};
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

/**
 * @brief Create the VDS sub-file path
 *
 * @param base_path The path to the vds sub-directory.
 * @param index_point The current leaf-task's index point.
 * @param store_shape The shape of the store being written.
 *
 * @return The sub-file name for the current leaf-task to write.
 */
[[nodiscard]] std::filesystem::path make_filepath(const std::filesystem::path& vds_dir,
                                                  const DomainPoint& index_point,
                                                  const Domain& store_shape)
{
  LEGATE_ASSERT(std::filesystem::is_directory(vds_dir));
  return vds_dir / fmt::format("{}_lo_{}_hi_{}.h5",
                               FlatDomainPoint{index_point},
                               FlatDomainPoint{store_shape.lo()},
                               FlatDomainPoint{store_shape.hi()});
}

/**
 * @brief The HDF5 write task common body.
 *
 * @param context The task context.
 * @param is_device `true` if the task is a GPU task, `false` otherwise.
 */
void task_body(const legate::TaskContext& context, bool is_device)
{
  auto&& index_point        = context.get_task_index();
  const auto store          = context.input(0).data();
  const auto base_dir       = std::filesystem::path{context.scalar(0).value<std::string_view>()};
  const auto dataset_name   = context.scalar(1).value<std::string>();
  const auto index_filepath = make_filepath(base_dir, index_point, store.domain());

  // The below is just an unrolled
  //
  // legate::double_dispatch(store.dim(), store.code(), HDF5ReadFn{}, ...);
  //
  // because double_dispatch() does not support Type::Code::BINARY yet, and it won't until
  // https://github.com/nv-legate/legate.internal/pull/1604 is resolved/merged.

#define TYPE_DISPATCH(__dim__)                                                               \
  case __dim__:                                                                              \
    TypeDispatcher<__dim__>{}(                                                               \
      store.code(), HDF5WriteFn{}, context, store, index_filepath, dataset_name, is_device); \
    break;

  switch (const auto dim = store.dim()) {
    LEGION_FOREACH_N(TYPE_DISPATCH);
    default: {  // legate-lint: no-switch-default
      legate::detail::throw_unsupported_dim(dim);
    }
  }

#undef TYPE_DISPATCH

  if (auto&& domain = context.get_launch_domain(); index_point == domain.lo()) {
    std::ofstream bounds_file{base_dir / "bounds.txt", std::ios::out | std::ios::trunc};

    bounds_file << fmt::format(
      "{}\n{}\n", FlatDomainPoint{domain.lo()}, FlatDomainPoint{domain.hi()});
  }
}

}  // namespace

/*static*/ void HDF5WriteVDS::cpu_variant(legate::TaskContext context)
{
  task_body(context, /* is_device */ false);
}

/*static*/ void HDF5WriteVDS::omp_variant(legate::TaskContext context) { cpu_variant(context); }

/*static*/ void HDF5WriteVDS::gpu_variant(legate::TaskContext context)
{
  task_body(context, /* is_device */ true);
}

}  // namespace legate::io::hdf5::detail
