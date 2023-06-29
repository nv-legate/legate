/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <gtest/gtest.h>
#include <fstream>

#include "cunumeric.h"
#include "legate.h"
#include "task_io.h"

namespace legateio {

class Array {
 public:
  Array(legate::LibraryContext* context, legate::LogicalStore store)
    : context_(context), store_(store)
  {
  }

  legate::LogicalStore store() { return store_; }

 protected:
  legate::LibraryContext* context_;
  legate::LogicalStore store_;
};

class IOArray : public Array {
 public:
  using Array::Array;

  void to_file(std::string filename)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto task    = runtime->create_task(context_, task::legateio::WRITE_FILE);
    auto part    = task.declare_partition();

    task.add_scalar_arg(legate::Scalar(filename));
    task.add_input(store_, part);

    // Request a broadcasting for the input. Since this is the only store
    // argument to the task, Legate will launch a single task from this
    // task descriptor.
    task.add_constraint(legate::broadcast(part, {0}));

    runtime->submit(std::move(task));
  }

  void to_uneven_tiles(std::string path)
  {
    int result = mkdir(path.c_str(), 0775);
    if (result == -1) {
      EXPECT_EQ(errno, EEXIST);
    } else {
      EXPECT_EQ(result, 0);
    }

    auto runtime = legate::Runtime::get_runtime();
    auto task    = runtime->create_task(context_, task::legateio::WRITE_UNEVEN_TILES);
    auto part    = task.declare_partition();

    task.add_scalar_arg(legate::Scalar(path));
    task.add_input(store_, part);
    runtime->submit(std::move(task));
  }

  void to_even_tiles(std::string path, uint32_t tile_shape)
  {
    int result = mkdir(path.c_str(), 0775);
    if (result == -1) {
      EXPECT_EQ(errno, EEXIST);
    } else {
      EXPECT_EQ(result, 0);
    }

    auto runtime         = legate::Runtime::get_runtime();
    auto store_partition = store_.partition_by_tiling({tile_shape, tile_shape});
    auto launch_shape    = store_partition.partition()->color_shape();

    auto task = runtime->create_task(context_, task::legateio::WRITE_EVEN_TILES, launch_shape);
    task.add_input(store_partition);
    task.add_scalar_arg(legate::Scalar(path));

    auto extents = store_.extents();
    EXPECT_EQ(extents.size(), 2);
    task.add_scalar_arg(
      legate::Scalar(std::vector<uint32_t>{(uint32_t)extents[0], (uint32_t)extents[1]}));
    task.add_scalar_arg(legate::Scalar(std::vector<uint32_t>{tile_shape, tile_shape}));
    runtime->submit(std::move(task));
  }

  static IOArray from_store(legate::LogicalStore store)
  {
    auto runtime = legate::Runtime::get_runtime();
    auto context = runtime->find_library(task::legateio::library_name);
    return IOArray(context, store);
  }
};

IOArray read_file(legate::LibraryContext* context,
                  std::string filename,
                  std::unique_ptr<legate::Type> dtype)
{
  auto runtime = legate::Runtime::get_runtime();
  auto output  = runtime->create_store(std::move(dtype), 1);
  auto task    = runtime->create_task(context, task::legateio::READ_FILE);
  auto part    = task.declare_partition();

  task.add_scalar_arg(legate::Scalar(filename));
  task.add_output(output, part);

  runtime->submit(std::move(task));
  return IOArray(context, output);
}

IOArray read_file_parallel(legate::LibraryContext* context,
                           std::string filename,
                           std::unique_ptr<legate::Type> dtype,
                           uint parallelism = 0)
{
  auto runtime = legate::Runtime::get_runtime();
  if (parallelism == 0) {
    parallelism = runtime->get_machine().count(legate::mapping::TaskTarget::CPU);
  }

  auto output = runtime->create_store(std::move(dtype), 1);
  auto task =
    runtime->create_task(context, task::legateio::READ_FILE, legate::Shape({parallelism}));

  task.add_scalar_arg(legate::Scalar(filename));
  task.add_output(output);

  runtime->submit(std::move(task));
  return IOArray(context, output);
}

void _read_header_uneven(std::string path, std::vector<size_t>& color_shape)
{
  std::ifstream in(path.c_str(), std::ios::binary | std::ios::in);

  uint32_t code;
  uint32_t dim;

  in.read(reinterpret_cast<char*>(&code), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

  EXPECT_EQ(code, static_cast<uint32_t>(legate::Type::Code::INT8));

  int64_t data;
  for (int i = 0; i < dim; i++) {
    in.read(reinterpret_cast<char*>(&data), sizeof(int64_t));
    color_shape.push_back(data);
  }
  in.close();
}

IOArray read_uneven_tiles(legate::LibraryContext* context, std::string path)
{
  // Read the dataset's header to find the type code and the color shape of
  // the partition, which are laid out in the header in the following way:
  //
  //  +-----------+--------+----------+-----
  //  | type code | # dims | extent 0 | ...
  //  |   (4B)    |  (4B)  |   (8B)   |
  //  +-----------+--------+----------+-----
  //
  std::vector<size_t> color_shape;
  _read_header_uneven(path + "/.header", color_shape);

  // Create a multi-dimensional unbound store
  //
  // Like 1D unbound stores, the shape of this store is also determined by
  // outputs from the tasks. For example, if the store is 2D and there are
  // four tasks (0, 0), (0, 1), (1, 0), and (1, 1) that respectively produce
  // 2x3, 2x4, 3x3, and 3x4 outputs, the store's shape would be (5, 7) and
  // internally partitioned in the following way:
  //
  //           0  1  2  3  4  5  6
  //         +--------+------------+
  //       0 | (0, 0) |   (0, 1)   |
  //       1 |        |            |
  //         +--------+------------+
  //       2 |        |            |
  //       3 | (1, 0) |   (1, 1)   |
  //       4 |        |            |
  //         +--------+------------+
  //
  auto runtime = legate::Runtime::get_runtime();
  auto output  = runtime->create_store(legate::int8(), color_shape.size());
  auto task =
    runtime->create_task(context, task::legateio::READ_UNEVEN_TILES, legate::Shape(color_shape));

  task.add_output(output);
  task.add_scalar_arg(legate::Scalar(path));
  runtime->submit(std::move(task));

  return IOArray(context, output);
}

void _read_header_even(std::string path,
                       std::vector<size_t>& shape,
                       std::vector<size_t>& tile_shape)
{
  std::ifstream in(path.c_str(), std::ios::binary | std::ios::in);

  uint32_t code;
  uint32_t dim;

  in.read(reinterpret_cast<char*>(&code), sizeof(uint32_t));
  in.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));

  EXPECT_EQ(code, static_cast<uint32_t>(legate::Type::Code::INT8));

  uint32_t data;
  for (int i = 0; i < dim; i++) {
    in.read(reinterpret_cast<char*>(&data), sizeof(uint32_t));
    shape.push_back(data);
  }
  for (int i = 0; i < dim; i++) {
    in.read(reinterpret_cast<char*>(&data), sizeof(uint32_t));
    tile_shape.push_back(data);
  }
  in.close();
}

IOArray read_even_tiles(legate::LibraryContext* context, std::string path)
{
  // Read the dataset's header to find the type code, the array's shape, and
  // the tile shape. The following shows the header's format:
  //
  //   +-----------+--------+----------+-----+----------+-----
  //   |           |        |  shape   |     |   tile   |
  //   | type code | # dims | extent 0 | ... | extent 0 | ...
  //   |   (4B)    |  (4B)  |   (4B)   |     |   (4B)   |
  //   +-----------+--------+----------+-----+----------+-----
  //
  std::vector<size_t> shape;
  std::vector<size_t> tile_shape;
  _read_header_even(path + "/.header", shape, tile_shape);

  auto runtime          = legate::Runtime::get_runtime();
  auto output           = runtime->create_store(shape, legate::int8());
  auto output_partition = output.partition_by_tiling(tile_shape);
  auto launch_shape     = output_partition.partition()->color_shape();
  auto task = runtime->create_task(context, task::legateio::READ_EVEN_TILES, launch_shape);

  task.add_output(output_partition);
  task.add_scalar_arg(legate::Scalar(path));
  runtime->submit(std::move(task));

  // Unlike AutoTask, manually parallelized tasks don't update the "key"
  // partition for their outputs.
  output.set_key_partition(runtime->get_machine(), output_partition.partition().get());

  return IOArray(context, output);
}

TEST(Example, SingleFileIO)
{
  legate::Core::perform_registration<task::legateio::register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(task::legateio::library_name);

  uint32_t n           = 10;
  std::string filename = "test.dat";

  auto src = cunumeric::arange(n).as_type(legate::int8());
  auto c1  = IOArray::from_store(src.get_store());

  // Dump the IOArray to a file
  c1.to_file(filename);

  // Issue an execution fence to make sure the writer task finishes before
  // any of the downstream tasks start.
  runtime->issue_execution_fence();

  // Read the file into a IOArray
  IOArray c2 = read_file(context, filename, legate::int8());
  EXPECT_TRUE(cunumeric::array_equal(src, cunumeric::as_array(c2.store())));

  // Read the file into a IOArray with a fixed degree of parallelism
  IOArray c3 = read_file_parallel(context, filename, legate::int8(), 2);
  EXPECT_TRUE(cunumeric::array_equal(src, cunumeric::as_array(c3.store())));

  // Read the file into a IOArray with the library-chosen degree of
  // parallelism
  IOArray c4 = read_file_parallel(context, filename, legate::int8());
  EXPECT_TRUE(cunumeric::array_equal(src, cunumeric::as_array(c4.store())));
}

TEST(Example, EvenTilesIO)
{
  legate::Core::perform_registration<task::legateio::register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(task::legateio::library_name);

  std::string dataset_name = "even_datafiles";
  uint32_t store_shape     = 8;
  uint32_t tile_shape      = 3;

  // Use cuNumeric to generate a random array to dump to a dataset
  auto src = cunumeric::random({store_shape, store_shape}).as_type(legate::int8());

  // Construct an IOArray from the cuNumeric ndarray
  auto c1 = IOArray::from_store(src.get_store());

  // Dump the IOArray to a dataset of even tiles
  c1.to_even_tiles(dataset_name, tile_shape);

  runtime->issue_execution_fence(true);

  // Read the dataset into an IOArray
  IOArray c2 = read_even_tiles(context, dataset_name);

  // Convert the IOArray into a cuNumeric ndarray and perform a binary
  // operation, just to confirm in the profile that the partition from the
  // reader tasks is reused in the downstream tasks.
  auto empty =
    cunumeric::full({store_shape, store_shape}, cunumeric::Scalar(static_cast<int64_t>(0)));
  auto c2_cunumeric =
    cunumeric::add(cunumeric::as_array(c2.store()).as_type(legate::int64()), empty);
  EXPECT_TRUE(cunumeric::array_equal(src, c2_cunumeric.as_type(legate::int8())));
}

TEST(Example, UnevenTilesIO)
{
  legate::Core::perform_registration<task::legateio::register_tasks>();

  auto runtime = legate::Runtime::get_runtime();
  auto context = runtime->find_library(task::legateio::library_name);

  std::string dataset_name = "uneven_datafiles";
  uint32_t store_shape     = 8;

  // Use cuNumeric to generate a random array to dump to a dataset
  auto src = cunumeric::random({store_shape, store_shape}).as_type(legate::int8());

  // Construct an IOArray from the cuNumeric ndarray
  auto c1 = IOArray::from_store(src.get_store());

  // Dump the IOArray to a dataset of even tiles
  c1.to_uneven_tiles(dataset_name);

  runtime->issue_execution_fence(true);

  // Read the dataset into an IOArray
  IOArray c2 = read_uneven_tiles(context, dataset_name);
  EXPECT_TRUE(cunumeric::array_equal(src, cunumeric::as_array(c2.store())));
}

}  // namespace legateio
