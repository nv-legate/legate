/* Copyright 2021 NVIDIA Corporation
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

#pragma once

#include <memory>

#include "core/utilities/tuple.h"
#include "legion.h"

namespace legate {

class LogicalStore;
class Projection;
class Runtime;

using Shape = tuple<size_t>;

struct Partition {
 public:
  enum class Kind : int32_t {
    NO_PARTITION = 0,
    TILING       = 1,
  };

 public:
  Partition(Runtime* runtime);
  virtual ~Partition() {}

 public:
  virtual Kind kind() const = 0;

 public:
  virtual bool is_complete_for(const LogicalStore* store) const = 0;
  virtual bool is_disjoint_for(const LogicalStore* store) const = 0;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool disjoint = false,
                                             bool complete = false) const      = 0;
  virtual std::unique_ptr<Projection> get_projection(LogicalStore store) const = 0;

 public:
  virtual bool has_launch_domain() const       = 0;
  virtual Legion::Domain launch_domain() const = 0;

 public:
  Runtime* runtime() const { return runtime_; }
  virtual std::string to_string() const = 0;

 protected:
  Runtime* runtime_;
};

class NoPartition : public Partition {
 public:
  NoPartition(Runtime* runtime);

 public:
  virtual Kind kind() const override { return Kind::NO_PARTITION; }

 public:
  virtual bool is_complete_for(const LogicalStore* store) const override;
  virtual bool is_disjoint_for(const LogicalStore* store) const override;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool disjoint,
                                             bool complete) const override;
  virtual std::unique_ptr<Projection> get_projection(LogicalStore store) const override;

 public:
  virtual bool has_launch_domain() const override;
  virtual Legion::Domain launch_domain() const override;

 public:
  virtual std::string to_string() const override;

 private:
  Runtime* runtime_;
};

class Tiling : public Partition {
 public:
  Tiling(Runtime* runtime, Shape&& tile_shape, Shape&& color_shape, Shape&& offsets);

 public:
  bool operator==(const Tiling& other) const;
  bool operator<(const Tiling& other) const;

 public:
  virtual Kind kind() const override { return Kind::TILING; }

 public:
  virtual bool is_complete_for(const LogicalStore* store) const override;
  virtual bool is_disjoint_for(const LogicalStore* store) const override;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool disjoint,
                                             bool complete) const override;
  virtual std::unique_ptr<Projection> get_projection(LogicalStore store) const override;

 public:
  virtual bool has_launch_domain() const override;
  virtual Legion::Domain launch_domain() const override;

 public:
  virtual std::string to_string() const override;

 public:
  const Shape& tile_shape() const { return tile_shape_; }
  const Shape& color_shape() const { return color_shape_; }
  const Shape& offsets() const { return offsets_; }

 private:
  Shape tile_shape_;
  Shape color_shape_;
  Shape offsets_;
};

std::unique_ptr<Partition> create_no_partition(Runtime* runtime);

std::unique_ptr<Partition> create_tiling(Runtime* runtime,
                                         Shape&& tile_shape,
                                         Shape&& color_shape,
                                         Shape&& offsets = {});

}  // namespace legate
