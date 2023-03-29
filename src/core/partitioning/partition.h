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

#include "core/data/shape.h"
#include "legion.h"

namespace legate {

class Projection;

namespace detail {

class LogicalStore;

}  // namespace detail

struct Partition {
 public:
  enum class Kind : int32_t {
    NO_PARTITION = 0,
    TILING       = 1,
  };

 public:
  Partition() {}
  virtual ~Partition() {}

 public:
  virtual Kind kind() const = 0;

 public:
  virtual bool is_complete_for(const detail::LogicalStore* store) const = 0;
  virtual bool is_disjoint_for(const detail::LogicalStore* store) const = 0;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool disjoint = false,
                                             bool complete = false) const       = 0;
  virtual std::unique_ptr<Projection> get_projection(detail::LogicalStore* store,
                                                     int32_t launch_ndim) const = 0;

 public:
  virtual bool has_launch_domain() const       = 0;
  virtual Legion::Domain launch_domain() const = 0;

 public:
  virtual std::unique_ptr<Partition> clone() const = 0;

 public:
  virtual std::string to_string() const = 0;
};

class NoPartition : public Partition {
 public:
  NoPartition();

 public:
  virtual Kind kind() const override { return Kind::NO_PARTITION; }

 public:
  virtual bool is_complete_for(const detail::LogicalStore* store) const override;
  virtual bool is_disjoint_for(const detail::LogicalStore* store) const override;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool disjoint,
                                             bool complete) const override;
  virtual std::unique_ptr<Projection> get_projection(detail::LogicalStore* store,
                                                     int32_t launch_ndim) const override;

 public:
  virtual bool has_launch_domain() const override;
  virtual Legion::Domain launch_domain() const override;

 public:
  virtual std::unique_ptr<Partition> clone() const override;

 public:
  virtual std::string to_string() const override;
};

class Tiling : public Partition {
 public:
  Tiling(Shape&& tile_shape, Shape&& color_shape, Shape&& offsets);

 public:
  Tiling(const Tiling&) = default;

 public:
  bool operator==(const Tiling& other) const;
  bool operator<(const Tiling& other) const;

 public:
  virtual Kind kind() const override { return Kind::TILING; }

 public:
  virtual bool is_complete_for(const detail::LogicalStore* store) const override;
  virtual bool is_disjoint_for(const detail::LogicalStore* store) const override;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool disjoint,
                                             bool complete) const override;
  virtual std::unique_ptr<Projection> get_projection(detail::LogicalStore* store,
                                                     int32_t launch_ndim) const override;

 public:
  virtual bool has_launch_domain() const override;
  virtual Legion::Domain launch_domain() const override;

 public:
  virtual std::unique_ptr<Partition> clone() const override;

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

std::unique_ptr<Partition> create_no_partition();

std::unique_ptr<Partition> create_tiling(Shape&& tile_shape,
                                         Shape&& color_shape,
                                         Shape&& offsets = {});

std::ostream& operator<<(std::ostream& out, const Partition& partition);

}  // namespace legate
