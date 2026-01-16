/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/interface.h>

#include <legate/data/scalar.h>
#include <legate/data/shape.h>
#include <legate/experimental/io/detail/library.h>
#include <legate/io/hdf5/detail/combine_vds.h>
#include <legate/io/hdf5/detail/hdf5_partitioner.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>
#include <legate/io/hdf5/detail/read.h>
#include <legate/io/hdf5/detail/write_vds.h>
#include <legate/io/hdf5/interface.h>
#include <legate/runtime/runtime.h>
#include <legate/tuning/scope.h>
#include <legate/type/types.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>

#include <fmt/format.h>
#include <fmt/std.h>

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace legate::io::hdf5::detail {

// ==========================================================================================

namespace {

[[nodiscard]] Type deduce_type_from_dataset(const wrapper::HDF5DataSet& dset)
{
  constexpr std::size_t _8_BIT  = 1;  // 1 byte
  constexpr std::size_t _16_BIT = 2;  // 2 bytes
  constexpr std::size_t _32_BIT = 4;  // 4 bytes
  constexpr std::size_t _64_BIT = 8;  // 8 bytes

  const auto dtype  = dset.type();
  const auto dclass = dtype.type_class();

  switch (dclass) {
    case wrapper::HDF5Type::Class::BOOL: return legate::bool_();
    case wrapper::HDF5Type::Class::SIGNED_INTEGER: {
      switch (const auto s = dtype.size()) {
        case _8_BIT: return legate::int8();
        case _16_BIT: return legate::int16();
        case _32_BIT: return legate::int32();
        case _64_BIT: return legate::int64();
        default:  // legate-lint: no-switch-default
          throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
            fmt::format("unhandled signed integer size: {}", s)};
      }
    }
    case wrapper::HDF5Type::Class::UNSIGNED_INTEGER: {
      switch (const auto s = dtype.size()) {
        case _8_BIT: return legate::uint8();
        case _16_BIT: return legate::uint16();
        case _32_BIT: return legate::uint32();
        case _64_BIT: return legate::uint64();
        default:  // legate-lint: no-switch-default
          throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
            fmt::format("unhandled unsigned integer size: {}", s)};
      }
    }
    case wrapper::HDF5Type::Class::FLOAT: {
      switch (const auto s = dtype.size()) {
        case _16_BIT: return legate::float16();
        case _32_BIT: return legate::float32();
        case _64_BIT: return legate::float64();
        default:  // legate-lint: no-switch-default
          throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
            fmt::format("unhandled floating point size: {}", s)};
      }
    }
    case wrapper::HDF5Type::Class::BITFIELD: [[fallthrough]];
    case wrapper::HDF5Type::Class::OPAQUE:
      return legate::binary_type(static_cast<std::uint32_t>(dtype.size()));
    case wrapper::HDF5Type::Class::STRING: return legate::string_type();
    // Unhandled types
    case wrapper::HDF5Type::Class::TIME: [[fallthrough]];
    case wrapper::HDF5Type::Class::COMPOUND: [[fallthrough]];
    case wrapper::HDF5Type::Class::REFERENCE: [[fallthrough]];
    case wrapper::HDF5Type::Class::ENUM: [[fallthrough]];
    case wrapper::HDF5Type::Class::VARIABLE_LENGTH: [[fallthrough]];
    case wrapper::HDF5Type::Class::ARRAY:
      throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
        fmt::format("unsupported HDF5 datatype: {}", dtype.to_string())};
  }
  LEGATE_ABORT("Unhandled HDF5 Datatype ", legate::detail::to_underlying(dclass));
}

[[nodiscard]] Shape deduce_shape_from_dataset(const wrapper::HDF5DataSet& dset)
{
  auto dims = dset.data_space().extents();

  if (dims.empty()) {
    // Must have dimension of at least 1, since the task does a dimension dispatch based off of
    // it, and dim = 0 is unsupported.
    dims.emplace_back(1);
  }
  return Shape{dims};
}

/**
 * @brief Create a LogicalArray from a HDF5 dataset and shape.
 *
 * @param dataset The HDF5 dataset.
 * @param shape The shape of the dataset.
 *
 * @return The LogicalArray.
 */
[[nodiscard]] LogicalArray create_output_array(const wrapper::HDF5DataSet& dataset,
                                               const Shape& shape)
{
  auto* rt              = Runtime::get_runtime();
  const auto array_type = deduce_type_from_dataset(dataset);

  // Safe to call volume here. We constructed the shape ourselves, so it is always ready and
  // volume() will never block.
  return rt->create_array(
    shape, array_type, /* nullable */ false, /* optimize_for_scalar */ shape.volume() <= 1);
}

/**
 * @brief Submit a tiled HDF5 read task and return the result array.
 *
 * Creates an output array, partitions it by the given tile shape, and submits
 * a parallel read task.
 *
 * @param native_path The path to the HDF5 file.
 * @param shape The shape of the dataset.
 * @param tile_shape The shape of each tile for partitioning.
 * @param dataset The dataset to read.
 *
 * @return The LogicalArray that will contain the read data.
 */
[[nodiscard]] LogicalArray submit_tiled_read_task(std::string_view native_path,
                                                  const Shape& shape,
                                                  Span<const std::uint64_t> tile_shape,
                                                  const wrapper::HDF5DataSet& dataset)
{
  auto ret       = create_output_array(dataset, shape);
  auto partition = ret.data().partition_by_tiling(tile_shape);
  auto* rt       = Runtime::get_runtime();

  // Create task with launch shape equal to the partition's color shape
  auto task = rt->create_task(experimental::io::detail::core_io_library(),
                              detail::HDF5Read::TASK_CONFIG.task_id(),
                              partition.color_shape());

  task.add_output(partition);
  task.add_scalar_arg(Scalar{native_path});
  task.add_scalar_arg(Scalar{dataset.name()});
  rt->submit(std::move(task));
  return ret;
}

}  // namespace

LogicalArray from_file(const std::filesystem::path& file_path, std::string_view dataset_name)
{
  if (!std::filesystem::exists(file_path)) {
    throw legate::detail::TracedException<std::system_error>{
      std::make_error_code(std::errc::no_such_file_or_directory), file_path};
  }

  auto&& native_path = file_path.native();
  auto* rt           = Runtime::get_runtime();
  const auto f       = wrapper::HDF5File{native_path, wrapper::HDF5File::OpenMode::READ_ONLY};

  if (!f.has_data_set(std::string{dataset_name})) {
    throw legate::detail::TracedException<InvalidDataSetError>{
      fmt::format("Dataset '{}' does not exist in {}", dataset_name, native_path),
      native_path,
      std::string{dataset_name}};
  }

  const auto dataset = f.data_set(std::string{dataset_name});
  const auto shape   = deduce_shape_from_dataset(dataset);

  const auto&& machine        = rt->get_machine();
  const auto& parallel_policy = Scope::parallel_policy();
  const std::size_t num_tiles =
    static_cast<std::size_t>(machine.count()) * parallel_policy.overdecompose_factor();

  // try to use a tiling strategy to match the layout of the dataset in files. If the dataset layout
  // is not supported, then we use an auto task, which will be parallelized automatically
  // by the runtime
  auto tile_shape = get_partition_tile_shape(shape, num_tiles, native_path, dataset);

  if (tile_shape) {
    return submit_tiled_read_task(native_path, shape, *tile_shape, dataset);
  }

  // this is the fallback case, we use an auto task, which will be parallelized automatically
  // by the runtime
  auto ret  = create_output_array(dataset, shape);
  auto task = rt->create_task(experimental::io::detail::core_io_library(),
                              detail::HDF5Read::TASK_CONFIG.task_id());

  task.add_scalar_arg(Scalar{native_path});
  task.add_scalar_arg(Scalar{dataset_name});
  task.add_output(ret);
  rt->submit(std::move(task));
  return ret;
}

namespace {

/**
 * @brief Computes the name of the directory containing the VDS subdirectories.
 *
 * If `base_path = /path/to/foo.h5` then returns `/path/to/foo_legate_vds`.
 *
 * @param base_path The base path of the resulting HDF5 file the user wants.
 *
 * @return The VDS subdirectory.
 */
[[nodiscard]] std::filesystem::path to_vds_dir(std::filesystem::path base_path)
{
  base_path.replace_filename(base_path.stem().native() + "_legate_vds");
  return base_path;
}

/**
 * @brief Normalizes a path.
 *
 * `path` need not exist. The parts of `path` that do exist will be fully resolved, by any
 * other parts will simply be normalized. So given `path = /path/exists/here/../does/not/exist`
 * will be normalized to `/path/exists/does/not/exist`.
 *
 * @param path The path to normalize.
 *
 * @return The canonical representation of the path.
 */
[[nodiscard]] std::filesystem::path normalize_path(const std::filesystem::path& path)
{
  // Can't use canonical because path may not exist yet.
  return std::filesystem::weakly_canonical(path).make_preferred();
}

}  // namespace

void to_file(const LogicalArray& array,
             std::filesystem::path file_path,
             std::string_view dataset_name)
{
  file_path = normalize_path(file_path);

  if (std::filesystem::is_directory(file_path)) {
    throw legate::detail::TracedException<std::invalid_argument>{
      fmt::format("File path ({}) must be the name of a file, not a directory", file_path)};
  }

  auto* const runtime     = Runtime::get_runtime();
  const auto vds_dir      = to_vds_dir(file_path);
  const auto vds_dir_scal = Scalar{vds_dir.native()};
  const auto dset_scal    = Scalar{dataset_name};

  // This dummy argument exists because the HDF5CombineVDS task requires that all the separate
  // VDS files have been written to disk first. We want to make it as small as possible because
  // this argument -- by virtue of being an input to a task with a launch domain of 1 -- will
  // be gathered to a single node by legion.
  const auto dummy_data_dependence = runtime->create_store(Scalar{bool{}});

  {
    auto task = runtime->create_task(experimental::io::detail::core_io_library(),
                                     detail::HDF5WriteVDS::TASK_CONFIG.task_id());

    task.add_scalar_arg(vds_dir_scal);
    task.add_scalar_arg(dset_scal);
    task.add_input(array);
    task.add_reduction(dummy_data_dependence, ReductionOpKind::ADD);
    // The point of no return. Once we submit the task, the user will be unable to potentially
    // catch any exceptions thrown by the task, so wait until this moment to actually create the
    // directories (because HDF5 will error if the base directory doesn't exist).
    std::filesystem::create_directories(vds_dir);
    runtime->submit(std::move(task));
  }

  auto task = runtime->create_task(experimental::io::detail::core_io_library(),
                                   detail::HDF5CombineVDS::TASK_CONFIG.task_id(),
                                   {1});

  task.add_scalar_arg(Scalar{file_path.native()});
  task.add_scalar_arg(vds_dir_scal);
  task.add_scalar_arg(Scalar{
    array.extents()  // This blocks
  });
  task.add_scalar_arg(dset_scal);
  task.add_input(dummy_data_dependence);

  runtime->submit(std::move(task));
}

}  // namespace legate::io::hdf5::detail
