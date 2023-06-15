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

#include "legion.h"

#include "core/data/shape.h"
#include "core/partitioning/restriction.h"
#include "core/utilities/typedefs.h"

namespace legate::detail {
class LogicalStore;
class Storage;
}  // namespace legate::detail

namespace legate {

struct Partition {
 public:
  enum class Kind : int32_t {
    NO_PARTITION = 0,
    TILING       = 1,
    WEIGHTED     = 2,
  };

 public:
  Partition() {}
  virtual ~Partition() {}

 public:
  virtual Kind kind() const = 0;

 public:
  virtual bool is_complete_for(const detail::Storage* storage) const          = 0;
  virtual bool is_disjoint_for(const Domain* launch_domain) const             = 0;
  virtual bool satisfies_restrictions(const Restrictions& restrictions) const = 0;
  virtual bool is_convertible() const                                         = 0;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool complete = false) const = 0;

 public:
  virtual bool has_launch_domain() const = 0;
  virtual Domain launch_domain() const   = 0;

 public:
  virtual std::unique_ptr<Partition> clone() const = 0;

 public:
  virtual std::string to_string() const = 0;

 public:
  virtual const Shape& color_shape() const = 0;
};

class NoPartition : public Partition {
 public:
  NoPartition();

 public:
  Kind kind() const override { return Kind::NO_PARTITION; }

 public:
  bool is_complete_for(const detail::Storage* storage) const override;
  bool is_disjoint_for(const Domain* launch_domain) const override;
  bool satisfies_restrictions(const Restrictions& restrictions) const override;
  bool is_convertible() const override { return true; }

 public:
  Legion::LogicalPartition construct(Legion::LogicalRegion region, bool complete) const override;

 public:
  bool has_launch_domain() const override;
  Domain launch_domain() const override;

 public:
  std::unique_ptr<Partition> clone() const override;

 public:
  std::string to_string() const override;

 public:
  const Shape& color_shape() const override
  {
    assert(false);
    throw std::invalid_argument("NoPartition doesn't support color_shape");
  }
};

class Tiling : public Partition {
 public:
  Tiling(Shape&& tile_shape, Shape&& color_shape, tuple<int64_t>&& offsets);

 public:
  Tiling(const Tiling&) = default;

 public:
  bool operator==(const Tiling& other) const;
  bool operator<(const Tiling& other) const;

 public:
  Kind kind() const override { return Kind::TILING; }

 public:
  bool is_complete_for(const detail::Storage* storage) const override;
  bool is_disjoint_for(const Domain* launch_domain) const override;
  bool satisfies_restrictions(const Restrictions& restrictions) const override;
  bool is_convertible() const override { return true; }

 public:
  Legion::LogicalPartition construct(Legion::LogicalRegion region, bool complete) const override;

 public:
  bool has_launch_domain() const override;
  Domain launch_domain() const override;

 public:
  std::unique_ptr<Partition> clone() const override;

 public:
  std::string to_string() const override;

 public:
  const Shape& tile_shape() const { return tile_shape_; }
  const Shape& color_shape() const override { return color_shape_; }
  const tuple<int64_t>& offsets() const { return offsets_; }

 public:
  Shape get_child_extents(const Shape& extents, const Shape& color);
  Shape get_child_offsets(const Shape& color);

 private:
  Shape tile_shape_;
  Shape color_shape_;
  tuple<int64_t> offsets_;
};

class Weighted : public Partition {
 public:
  Weighted(const Legion::FutureMap& weights, const Domain& color_domain);

 public:
  Weighted(const Weighted&) = default;

 public:
  bool operator==(const Weighted& other) const;
  bool operator<(const Weighted& other) const;

 public:
  Kind kind() const override { return Kind::WEIGHTED; }

 public:
  bool is_complete_for(const detail::Storage* storage) const override;
  bool is_disjoint_for(const Domain* launch_domain) const override;
  bool satisfies_restrictions(const Restrictions& restrictions) const override;
  bool is_convertible() const override { return false; }

 public:
  Legion::LogicalPartition construct(Legion::LogicalRegion region, bool complete) const override;

 public:
  bool has_launch_domain() const override;
  Domain launch_domain() const override;

 public:
  std::unique_ptr<Partition> clone() const override;

 public:
  std::string to_string() const override;

 public:
  const Shape& color_shape() const override { return color_shape_; }

 private:
  Legion::FutureMap weights_;
  Domain color_domain_;
  Shape color_shape_;
};

std::unique_ptr<NoPartition> create_no_partition();

std::unique_ptr<Tiling> create_tiling(Shape&& tile_shape,
                                      Shape&& color_shape,
                                      tuple<int64_t>&& offsets = {});

std::unique_ptr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                          const Domain& color_domain);

std::ostream& operator<<(std::ostream& out, const Partition& partition);

}  // namespace legate
