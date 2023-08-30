/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
    IMAGE        = 3,
  };

 public:
  Partition()                                = default;
  virtual ~Partition()                       = default;
  Partition(const Partition&)                = default;
  Partition(Partition&&) noexcept            = default;
  Partition& operator=(const Partition&)     = default;
  Partition& operator=(Partition&&) noexcept = default;

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

class Image : public Partition {
 public:
  Image(std::shared_ptr<detail::LogicalStore> func, std::shared_ptr<Partition> func_partition);

 public:
  Image(const Image&) = default;

 public:
  bool operator==(const Image& other) const;
  bool operator<(const Image& other) const;

 public:
  Kind kind() const override { return Kind::IMAGE; }

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
  const Shape& color_shape() const override;

 private:
  std::shared_ptr<detail::LogicalStore> func_;
  std::shared_ptr<Partition> func_partition_;
};

std::unique_ptr<NoPartition> create_no_partition();

std::unique_ptr<Tiling> create_tiling(Shape&& tile_shape,
                                      Shape&& color_shape,
                                      tuple<int64_t>&& offsets = {});

std::unique_ptr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                          const Domain& color_domain);

std::unique_ptr<Image> create_image(std::shared_ptr<detail::LogicalStore> func,
                                    std::shared_ptr<Partition> func_partition);

std::ostream& operator<<(std::ostream& out, const Partition& partition);

}  // namespace legate
