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

#include <legate/data/scalar.h>
#include <legate/data/shape.h>
#include <legate/experimental/io/detail/library.h>
#include <legate/io/hdf5/detail/read.h>
#include <legate/io/hdf5/detail/util.h>
#include <legate/io/hdf5/interface.h>
#include <legate/runtime/runtime.h>
#include <legate/type/type_info.h>
#include <legate/utilities/detail/traced_exception.h>

#include <highfive/H5DataSet.hpp>

#include <filesystem>
#include <fmt/format.h>
#include <stdexcept>
#include <string_view>
#include <system_error>

namespace legate::io::hdf5 {

InvalidDataSetError::InvalidDataSetError(const std::string& what,
                                         std::filesystem::path path,
                                         std::string dataset_name)
  : invalid_argument{what}, path_{std::move(path)}, dataset_name_{std::move(dataset_name)}
{
}

const std::filesystem::path& InvalidDataSetError::path() const noexcept { return path_; }

std::string_view InvalidDataSetError::dataset_name() const noexcept { return dataset_name_; }

// ==========================================================================================

namespace {

[[nodiscard]] Type deduce_type_from_dataset(const detail::HDF5GlobalLock&,
                                            const HighFive::DataSet& dset)
{
  constexpr std::size_t _8_BIT  = 1;  // 1 byte
  constexpr std::size_t _16_BIT = 2;  // 2 bytes
  constexpr std::size_t _32_BIT = 4;  // 4 bytes
  constexpr std::size_t _64_BIT = 8;  // 8 bytes

  auto&& dtype = dset.getDataType();

  switch (dtype.getClass()) {
    case HighFive::DataTypeClass::Integer: {
      switch (const auto s = dtype.getSize()) {
        case _8_BIT: return legate::int8();
        case _16_BIT: return legate::int16();
        case _32_BIT: return legate::int32();
        case _64_BIT: return legate::int64();
        default:
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
        default:
          throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
            fmt::format("unhandled floating point size: {}", s)};
      }
    }
    case HighFive::DataTypeClass::BitField:
      return legate::binary_type(static_cast<std::uint32_t>(dtype.getSize()));
    case HighFive::DataTypeClass::String: return legate::string_type();
    case HighFive::DataTypeClass::Invalid: return legate::null_type();
    default:
      throw legate::detail::TracedException<UnsupportedHDF5DataTypeError>{
        fmt::format("unsupported HDF5 datatype: {}", dtype.string())};
  }
}

[[nodiscard]] Shape deduce_shape_from_dataset(const detail::HDF5GlobalLock&,
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
  return Shape{std::move(uint_dims)};
}

[[nodiscard]] LogicalArray create_output_array(const std::string& path,
                                               std::string_view dataset_name,
                                               Runtime* rt)
{
  const detail::HDF5GlobalLock lock{};
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

  auto task =
    rt->create_task(experimental::io::detail::core_io_library(), detail::HDF5Read::TASK_ID);

  task.add_scalar_arg(Scalar{native_path});
  task.add_scalar_arg(Scalar{dataset_name});
  task.add_output(ret);
  task.set_side_effect(true);
  rt->submit(std::move(task));
  return ret;
}

}  // namespace legate::io::hdf5
