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

#include "core/data/shape.h"
#include "core/mapping/detail/machine.h"
#include "core/partitioning/restriction.h"
#include "core/utilities/typedefs.h"

#include <iosfwd>
#include <memory>
#include <string>

namespace legate::detail {
class LogicalStore;
class Storage;
}  // namespace legate::detail

namespace legate {

struct Partition {
 public:
  enum class Kind : int32_t {
    NO_PARTITION,
    TILING,
    WEIGHTED,
    IMAGE,
  };

  Partition()                                = default;
  virtual ~Partition()                       = default;
  Partition(const Partition&)                = default;
  Partition(Partition&&) noexcept            = default;
  Partition& operator=(const Partition&)     = default;
  Partition& operator=(Partition&&) noexcept = default;

  [[nodiscard]] virtual Kind kind() const = 0;

  [[nodiscard]] virtual bool is_complete_for(const detail::Storage* storage) const          = 0;
  [[nodiscard]] virtual bool is_disjoint_for(const Domain& launch_domain) const             = 0;
  [[nodiscard]] virtual bool satisfies_restrictions(const Restrictions& restrictions) const = 0;
  [[nodiscard]] virtual bool is_convertible() const                                         = 0;

  [[nodiscard]] virtual std::unique_ptr<Partition> scale(const Shape& factors) const      = 0;
  [[nodiscard]] virtual std::unique_ptr<Partition> bloat(const Shape& low_offsets,
                                                         const Shape& high_offsets) const = 0;

  [[nodiscard]] virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                           bool complete = false) const = 0;

  [[nodiscard]] virtual bool has_launch_domain() const = 0;
  [[nodiscard]] virtual Domain launch_domain() const   = 0;

  [[nodiscard]] virtual std::unique_ptr<Partition> clone() const = 0;

  [[nodiscard]] virtual std::string to_string() const = 0;

  [[nodiscard]] virtual const Shape& color_shape() const = 0;
};

class NoPartition : public Partition {
 public:
  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage* /*storage*/) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& /*restrictions*/) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] std::unique_ptr<Partition> scale(const Shape& factors) const override;
  [[nodiscard]] std::unique_ptr<Partition> bloat(const Shape& low_offsets,
                                                 const Shape& high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion /*region*/,
                                                   bool /*complete*/) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::unique_ptr<Partition> clone() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Shape& color_shape() const override;
};

class Tiling : public Partition {
 public:
  Tiling(Shape&& tile_shape, Shape&& color_shape, tuple<int64_t>&& offsets);
  Tiling(Shape&& tile_shape, Shape&& color_shape, tuple<int64_t>&& offsets, Shape&& strides);

  bool operator==(const Tiling& other) const;
  bool operator<(const Tiling& other) const;

  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage* storage) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& restrictions) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] std::unique_ptr<Partition> scale(const Shape& factors) const override;
  [[nodiscard]] std::unique_ptr<Partition> bloat(const Shape& low_offsets,
                                                 const Shape& high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::unique_ptr<Partition> clone() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Shape& tile_shape() const;
  [[nodiscard]] const Shape& color_shape() const override;
  [[nodiscard]] const tuple<int64_t>& offsets() const;

  [[nodiscard]] Shape get_child_extents(const Shape& extents, const Shape& color) const;
  [[nodiscard]] Shape get_child_offsets(const Shape& color) const;

 private:
  bool overlapped_{};
  Shape tile_shape_{};
  Shape color_shape_{};
  tuple<int64_t> offsets_{};
  Shape strides_{};
};

class Weighted : public Partition {
 public:
  Weighted(const Legion::FutureMap& weights, const Domain& color_domain);
  ~Weighted() override;

  Weighted(const Weighted&);

  bool operator==(const Weighted& other) const;
  bool operator<(const Weighted& other) const;

  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& restrictions) const override;
  [[nodiscard]] bool is_convertible() const override;
  [[nodiscard]] bool is_complete_for(const detail::Storage* /*storage*/) const override;

  [[nodiscard]] std::unique_ptr<Partition> scale(const Shape& factors) const override;
  [[nodiscard]] std::unique_ptr<Partition> bloat(const Shape& low_offsets,
                                                 const Shape& high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::unique_ptr<Partition> clone() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Shape& color_shape() const override;

 private:
  std::unique_ptr<Legion::FutureMap> weights_{nullptr};
  Domain color_domain_{};
  Shape color_shape_{};
};

class Image : public Partition {
 public:
  Image(std::shared_ptr<detail::LogicalStore> func,
        std::shared_ptr<Partition> func_partition,
        mapping::detail::Machine machine);

  bool operator==(const Image& other) const;
  bool operator<(const Image& other) const;

  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage* storage) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& restrictions) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] std::unique_ptr<Partition> scale(const Shape& factors) const override;
  [[nodiscard]] std::unique_ptr<Partition> bloat(const Shape& low_offsets,
                                                 const Shape& high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::unique_ptr<Partition> clone() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const Shape& color_shape() const override;

 private:
  std::shared_ptr<detail::LogicalStore> func_;
  std::shared_ptr<Partition> func_partition_{};
  mapping::detail::Machine machine_{};
};

[[nodiscard]] std::unique_ptr<NoPartition> create_no_partition();

[[nodiscard]] std::unique_ptr<Tiling> create_tiling(Shape&& tile_shape,
                                                    Shape&& color_shape,
                                                    tuple<int64_t>&& offsets = {});

[[nodiscard]] std::unique_ptr<Tiling> create_tiling(Shape&& tile_shape,
                                                    Shape&& color_shape,
                                                    tuple<int64_t>&& offsets,
                                                    Shape&& strides);

[[nodiscard]] std::unique_ptr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                                        const Domain& color_domain);

[[nodiscard]] std::unique_ptr<Image> create_image(std::shared_ptr<detail::LogicalStore> func,
                                                  std::shared_ptr<Partition> func_partition,
                                                  mapping::detail::Machine machine);

std::ostream& operator<<(std::ostream& out, const Partition& partition);

}  // namespace legate

#include "core/partitioning/partition.inl"
