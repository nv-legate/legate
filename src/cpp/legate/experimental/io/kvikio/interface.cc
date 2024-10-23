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

#include <legate/data/shape.h>
#include <legate/experimental/io/detail/library.h>
#include <legate/experimental/io/kvikio/detail/basic.h>
#include <legate/experimental/io/kvikio/detail/tile.h>
#include <legate/experimental/io/kvikio/detail/tile_by_offsets.h>
#include <legate/experimental/io/kvikio/interface.h>
#include <legate/runtime/runtime.h>
#include <legate/type/type_info.h>
#include <legate/utilities/detail/zip.h>

#include <cstdint>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <fstream>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <vector>

namespace legate::experimental::io::kvikio {

namespace {

void check_file_exists(const std::filesystem::path& path)
{
  if (!std::filesystem::exists(path)) {
    throw std::system_error{std::make_error_code(std::errc::no_such_file_or_directory), path};
  }
}

}  // namespace

// TODO (jfaibussowit):
// Don't pass require passing type, we should be able to deduce the datatype somehow.
LogicalArray from_file(const std::filesystem::path& file_path, const Type& type)
{
  check_file_exists(file_path);

  auto* rt = Runtime::get_runtime();
  // The task assumes the file is laid out as one giant linear array in memory. Therefore we
  // can deduce the size of the required store by checking the size of the file itself.
  //
  // This is horribly error-prone and relies on several properties:
  // 1. The filesystem includes no extra header or footer in the file data itself, i.e. that
  //    the size of the file is the size of the array.
  // 2. The filesystem stores the data in a way that it suitably aligned for the datatype,
  //    i.e. that the task has to do no pointer swizzling to align the pointers up.
  // 3. That the file really does contain the datatype in question. Inside the task we see only
  //    bytes, and effectively do memcpy(buf, file_data, file_size). This means that absolutely
  //    no conversions are done.
  //
  // We really should not do this.
  const auto array_size = std::filesystem::file_size(file_path) / type.size();
  auto ret              = rt->create_array(Shape{static_cast<std::uint64_t>(array_size)}, type);
  auto task = rt->create_task(io::detail::core_io_library(), detail::BasicRead::TASK_ID);

  task.add_scalar_arg(Scalar{file_path.native()});
  task.add_output(ret);
  task.set_side_effect(true);
  rt->submit(std::move(task));
  return ret;
}

namespace {

void clear_file(const std::filesystem::path& path)
{
  // touch the file to create an empty one
  const auto touch = std::fstream{path, std::ios::trunc | std::ios::out};

  static_cast<void>(touch);
}

}  // namespace

void to_file(const std::filesystem::path& file_path, const LogicalArray& array)
{
  if (const auto dim = array.dim(); dim != 1) {
    throw std::invalid_argument{fmt::format("number of array dimensions must be 1 (have {})", dim)};
  }

  auto* rt  = Runtime::get_runtime();
  auto task = rt->create_task(io::detail::core_io_library(), detail::BasicWrite::TASK_ID);

  task.add_scalar_arg(Scalar{file_path.native()});
  task.add_input(array);
  task.set_side_effect(true);

  // Truncate the file because each leaf task opens the file in "r+" mode (because otherwise
  // each open() of the file would overwrite it).
  //
  // We do this right before submission to ensure that we do as much error checking as possible
  // before doing filesystem modifications.
  //
  // Note: the existence of any exceptions being thrown from this function is not advertised to
  // the user, because the fact that we truncate the file is not really a user-visible
  // side-effect of this function. In fact, it would be confusing, because why would we
  // truncate a file if we are going to write to it??
  clear_file(file_path);

  rt->submit(std::move(task));
}

// ==========================================================================================

namespace {

void sanity_check_sizes(const LogicalArray& array,
                        const std::vector<std::uint64_t>& tile_shape,
                        const std::vector<std::uint64_t>& tile_start)
{
  if (tile_start.size() != tile_shape.size()) {
    throw std::invalid_argument{
      fmt::format("tile_start and tile_shape must have the same size. tile_start.size() = {}, "
                  "tile_shape.size() = {}",
                  tile_start.size(),
                  tile_shape.size())};
  }

  if (array.dim() != tile_shape.size()) {
    throw std::invalid_argument{
      fmt::format("Array and tile_shape must have the same size. Array.dim() = {}, "
                  "tile_shape.size() = {}",
                  array.dim(),
                  tile_shape.size())};
  }

  {
    auto extents = array.shape().extents();

    for (auto&& [d, c] : legate::detail::zip_equal(extents, tile_shape)) {
      if (d % c != 0) {
        throw std::invalid_argument{fmt::format(
          "The array shape ({}) must be divisible by the tile shape ({})", extents, tile_shape)};
      }
    }
  }
}

}  // namespace

// TODO (jfaibussowit):
// Don't pass require passing type, we should be able to deduce the datatype somehow.
LogicalArray from_file(const std::filesystem::path& file_path,
                       const Shape& shape,
                       const Type& type,
                       const std::vector<std::uint64_t>& tile_shape,
                       std::optional<std::vector<std::uint64_t>> tile_start)
{
  check_file_exists(file_path);

  auto* rt = Runtime::get_runtime();
  auto ret = rt->create_array(shape, type);

  if (!tile_start.has_value()) {
    // () ctor is deliberate here, we want a vector of 0's like tile_shape
    tile_start = std::vector<std::uint64_t>(tile_shape.size(), 0);
  }
  sanity_check_sizes(ret, tile_shape, *tile_start);

  auto partition = ret.data().partition_by_tiling(tile_shape);
  auto task      = rt->create_task(
    io::detail::core_io_library(), detail::TileRead::TASK_ID, partition.color_shape());

  task.add_output(partition);
  task.add_scalar_arg(Scalar{file_path.native()});
  task.add_scalar_arg(Scalar{*tile_start});
  rt->submit(std::move(task));
  return ret;
}

void to_file(const std::filesystem::path& file_path,
             const LogicalArray& array,
             const std::vector<std::uint64_t>& tile_shape,
             std::optional<std::vector<std::uint64_t>> tile_start)
{
  if (!tile_start.has_value()) {
    // () ctor is deliberate here, we want a vector of 0's like tile_shape
    tile_start = std::vector<std::uint64_t>(tile_shape.size(), 0);
  }
  sanity_check_sizes(array, tile_shape, *tile_start);

  auto* rt       = Runtime::get_runtime();
  auto partition = array.data().partition_by_tiling(tile_shape);
  auto task      = rt->create_task(
    io::detail::core_io_library(), detail::TileWrite::TASK_ID, partition.color_shape());

  task.add_input(partition);
  task.add_scalar_arg(Scalar{file_path.native()});
  task.add_scalar_arg(Scalar{*tile_start});
  rt->submit(std::move(task));
}

// ==========================================================================================

LogicalArray from_file_by_offsets(const std::filesystem::path& file_path,
                                  const Shape& shape,
                                  const Type& type,
                                  const std::vector<std::uint64_t>& offsets,
                                  const std::vector<std::uint64_t>& tile_shape)
{
  check_file_exists(file_path);

  auto* rt            = Runtime::get_runtime();
  auto ret            = rt->create_array(shape, type);
  auto partition      = ret.data().partition_by_tiling(tile_shape);
  auto&& launch_shape = partition.color_shape();

  if (const auto launch_vol = launch_shape.volume(); launch_vol != offsets.size()) {
    throw std::invalid_argument{
      fmt::format("Number of offsets ({}) must match the number of array tiles ({})",
                  offsets.size(),
                  launch_vol)};
  }

  auto task = rt->create_task(
    io::detail::core_io_library(), detail::TileByOffsetsRead::TASK_ID, launch_shape);

  task.add_output(partition);
  task.add_scalar_arg(Scalar{file_path.native()});
  task.add_scalar_arg(Scalar{offsets});
  rt->submit(std::move(task));
  return ret;
}

}  // namespace legate::experimental::io::kvikio
