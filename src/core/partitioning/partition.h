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

namespace legate {

class Projection;

namespace detail {

class LogicalStore;
class Storage;

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
  virtual bool is_complete_for(const detail::Storage* storage) const          = 0;
  virtual bool is_disjoint_for(const Domain* launch_domain) const             = 0;
  virtual bool satisfies_restrictions(const Restrictions& restrictions) const = 0;

 public:
  virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                             bool complete = false) const = 0;

 public:
  virtual bool has_launch_domain() const       = 0;
  virtual Legion::Domain launch_domain() const = 0;

 public:
  virtual std::unique_ptr<Partition> clone() const = 0;

 public:
  virtual std::string to_string() const = 0;

 public:
  virtual const Shape& tile_shape() const       = 0;
  virtual const Shape& color_shape() const      = 0;
  virtual const tuple<int64_t>& offsets() const = 0;
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

 public:
  Legion::LogicalPartition construct(Legion::LogicalRegion region, bool complete) const override;

 public:
  bool has_launch_domain() const override;
  Legion::Domain launch_domain() const override;

 public:
  std::unique_ptr<Partition> clone() const override;

 public:
  std::string to_string() const override;

 public:
  const Shape& tile_shape() const override
  {
    throw std::invalid_argument("Partition kind doesn't support tile_shape");
  }
  const Shape& color_shape() const override
  {
    throw std::invalid_argument("Partition kind doesn't support color_shape");
  }
  const tuple<int64_t>& offsets() const override
  {
    throw std::invalid_argument("Partition kind doesn't support offsets");
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

 public:
  Legion::LogicalPartition construct(Legion::LogicalRegion region, bool complete) const override;

 public:
  bool has_launch_domain() const override;
  Legion::Domain launch_domain() const override;

 public:
  std::unique_ptr<Partition> clone() const override;

 public:
  std::string to_string() const override;

 public:
  const Shape& tile_shape() const override { return tile_shape_; }
  const Shape& color_shape() const override { return color_shape_; }
  const tuple<int64_t>& offsets() const override { return offsets_; }

 public:
  Shape get_child_extents(const Shape& extents, const Shape& color);
  Shape get_child_offsets(const Shape& color);

 private:
  Shape tile_shape_;
  Shape color_shape_;
  tuple<int64_t> offsets_;
};

std::unique_ptr<Partition> create_no_partition();

std::unique_ptr<Partition> create_tiling(Shape&& tile_shape,
                                         Shape&& color_shape,
                                         tuple<int64_t>&& offsets = {});

std::ostream& operator<<(std::ostream& out, const Partition& partition);

}  // namespace legate
