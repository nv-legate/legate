/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/io/hdf5/detail/interface.h>

#include <legate/data/scalar.h>
#include <legate/data/shape.h>
#include <legate/experimental/io/detail/library.h>
#include <legate/io/hdf5/detail/combine_vds.h>
#include <legate/io/hdf5/detail/hdf5_wrapper.h>
#include <legate/io/hdf5/detail/read.h>
#include <legate/io/hdf5/detail/util.h>
#include <legate/io/hdf5/detail/write_vds.h>
#include <legate/io/hdf5/interface.h>
#include <legate/runtime/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/type_traits.h>

#include <highfive/H5DataSet.hpp>

#include <fmt/format.h>
#include <fmt/std.h>

#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace legate::io::hdf5::detail {

// ==========================================================================================

namespace {

[[nodiscard]] Type deduce_type_from_dataset(const detail::wrapper::HDF5MaybeLockGuard&,
                                            const HighFive::DataSet& dset)
{
  constexpr std::size_t _8_BIT  = 1;  // 1 byte
  constexpr std::size_t _16_BIT = 2;  // 2 bytes
  constexpr std::size_t _32_BIT = 4;  // 4 bytes
  constexpr std::size_t _64_BIT = 8;  // 8 bytes

  auto&& dtype      = dset.getDataType();
  const auto dclass = dtype.getClass();

  switch (dclass) {
    case HighFive::DataTypeClass::Integer: {
      switch (const auto s = dtype.getSize()) {
        case _8_BIT: return legate::int8();
        case _16_BIT: return legate::int16();
        case _32_BIT: return legate::int32();
        case _64_BIT: return legate::int64();
        default:  // legate-lint: no-switch-default
          throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
            fmt::format("unhandled integer size: {}", s)};
      }
    }
    case HighFive::DataTypeClass::Float: {
      switch (const auto s = dtype.getSize()) {
        case _16_BIT:
          // HighFive throws "Type given to create_and_check_datatype is not valid" if you try
          // to construct a float16. I suppose we could just let it through (in the hopes that
          // eventually they do support it), but for now we catch this explicitly.
          throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{fmt::format(
            "unsupported floating point size: {}. Legate supports this datatype but HDF5 does "
            "not",
            s)};
        case _32_BIT: return legate::float32();
        case _64_BIT: return legate::float64();
        default:  // legate-lint: no-switch-default
          throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
            fmt::format("unhandled floating point size: {}", s)};
      }
    }
    case HighFive::DataTypeClass::BitField: [[fallthrough]];
    case HighFive::DataTypeClass::Opaque:
      return legate::binary_type(static_cast<std::uint32_t>(dtype.getSize()));
    case HighFive::DataTypeClass::String: return legate::string_type();
    case HighFive::DataTypeClass::Invalid: {
      return legate::null_type();
    }
      // Unhandled types
    case HighFive::DataTypeClass::Time: [[fallthrough]];
    case HighFive::DataTypeClass::Compound: [[fallthrough]];
    case HighFive::DataTypeClass::Reference: [[fallthrough]];
    case HighFive::DataTypeClass::Enum: [[fallthrough]];
    case HighFive::DataTypeClass::VarLen: [[fallthrough]];
    case HighFive::DataTypeClass::Array:
      throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
        fmt::format("unsupported HDF5 datatype: {}", dtype.string())};
  }
  LEGATE_ABORT("Unhandled HDF5 Datatype ", legate::detail::to_underlying(dclass));
}

[[nodiscard]] Shape deduce_shape_from_dataset(const detail::wrapper::HDF5MaybeLockGuard&,
                                              const HighFive::DataSet& dset)
{
  auto uint_dims = [&]() -> std::vector<std::uint64_t> {
    auto&& dims = dset.getDimensions();

    return {dims.begin(), dims.end()};
  }();

  if (uint_dims.empty()) {
    // Must have dimension of at least 1, since the task does a dimension dispatch based off of
    // it, and dim = 0 is unsupported.
    uint_dims.emplace_back(1);
  }
  return Shape{uint_dims};
}

[[nodiscard]] LogicalArray create_output_array(const std::string& path,
                                               std::string_view dataset_name,
                                               Runtime* rt)
{
  const detail::wrapper::HDF5MaybeLockGuard lock{};
  const auto f       = detail::open_hdf5_file(lock, path, /*gds*/ false);
  const auto dataset = [&](const std::string& name) {
    if (!f.exist(name) || (f.getObjectType(name) != HighFive::ObjectType::Dataset)) {
      throw legate::detail::TracedException<InvalidDataSetError>{
        fmt::format("Dataset '{}' does not exist in {}", name, path), path, name};
    }
    return f.getDataSet(name);
  }(std::string{dataset_name});
  const auto shape      = deduce_shape_from_dataset(lock, dataset);
  const auto array_type = deduce_type_from_dataset(lock, dataset);

  // Safe to call volume here. We constructed the shape ourselves, so it is always ready and
  // volume() will never block.
  return rt->create_array(
    shape, array_type, /* nullable */ false, /* optimize_for_scalar */ shape.volume() <= 1);
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
  auto ret           = create_output_array(native_path, dataset_name, rt);

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

  {
    auto task = runtime->create_task(experimental::io::detail::core_io_library(),
                                     detail::HDF5WriteVDS::TASK_CONFIG.task_id());

    task.add_scalar_arg(vds_dir_scal);
    task.add_scalar_arg(dset_scal);
    task.add_input(array);
    // The point of no return. Once we submit the task, the user will be unable to potentially
    // catch any exceptions thrown by the task, so wait until this moment to actually create the
    // directories (because HDF5 will error if the base directory doesn't exist).
    std::filesystem::create_directories(vds_dir);
    runtime->submit(std::move(task));
  }

  // NOTE(jfaibussowit)
  // We must issue an execution fence here because the subsequent stitching task requires that
  // all the separate VDS files have been written to disk first. *Technically* we could enforce
  // this with a dummy data-dependency, but because the below task is a singleton task, this
  // won't work in streaming scopes (which currently still require all tasks to have the same
  // launch domains).
  //
  // An alternative would be to only issue the fence if we're in a streaming scope (and use
  // data dependency otherwise), but this seems like a big change to the semantics of the
  // task. It would be weird then to a future reader to see the "input" inside the task but
  // never use it.
  //
  // Another alternative would be to make the stitching task be an index launch where all index
  // points except the first do nothing. But that's very inefficient.
  //
  // It's not clear to me what the best solution here is.
  runtime->issue_execution_fence();

  auto task = runtime->create_task(experimental::io::detail::core_io_library(),
                                   detail::HDF5CombineVDS::TASK_CONFIG.task_id(),
                                   {1});

  task.add_scalar_arg(Scalar{file_path.native()});
  task.add_scalar_arg(vds_dir_scal);
  task.add_scalar_arg(Scalar{
    array.extents()  // This blocks
  });
  task.add_scalar_arg(dset_scal);

  runtime->submit(std::move(task));
}

}  // namespace legate::io::hdf5::detail
