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
#include "core/utilities/detail/hash.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/typedefs.h"

#include <iosfwd>
#include <memory>
#include <string>

namespace legate::detail {
class LogicalStore;
class Storage;
}  // namespace legate::detail

namespace legate {

class Partition {
 public:
  enum class Kind : std::uint8_t {
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

  [[nodiscard]] virtual std::unique_ptr<Partition> scale(
    const tuple<std::uint64_t>& factors) const = 0;
  [[nodiscard]] virtual std::unique_ptr<Partition> bloat(
    const tuple<std::uint64_t>& low_offsets, const tuple<std::uint64_t>& high_offsets) const = 0;

  // NOLINTNEXTLINE(google-default-arguments)
  [[nodiscard]] virtual Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                           bool complete = false) const = 0;

  [[nodiscard]] virtual bool has_launch_domain() const = 0;
  [[nodiscard]] virtual Domain launch_domain() const   = 0;

  [[nodiscard]] virtual std::unique_ptr<Partition> clone() const = 0;

  [[nodiscard]] virtual std::string to_string() const = 0;

  [[nodiscard]] virtual const tuple<std::uint64_t>& color_shape() const = 0;
};

class NoPartition : public Partition {
 public:
  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage* /*storage*/) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& /*restrictions*/) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] std::unique_ptr<Partition> scale(
    const tuple<std::uint64_t>& factors) const override;
  [[nodiscard]] std::unique_ptr<Partition> bloat(
    const tuple<std::uint64_t>& low_offsets,
    const tuple<std::uint64_t>& high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion /*region*/,
                                                   bool /*complete*/) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::unique_ptr<Partition> clone() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const tuple<std::uint64_t>& color_shape() const override;
};

class Tiling : public Partition {
 public:
  Tiling(tuple<std::uint64_t>&& tile_shape,
         tuple<std::uint64_t>&& color_shape,
         tuple<std::int64_t>&& offsets);
  Tiling(tuple<std::uint64_t>&& tile_shape,
         tuple<std::uint64_t>&& color_shape,
         tuple<std::int64_t>&& offsets,
         tuple<std::uint64_t>&& strides);

  bool operator==(const Tiling& other) const;

  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage* storage) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& restrictions) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] std::unique_ptr<Partition> scale(
    const tuple<std::uint64_t>& factors) const override;
  [[nodiscard]] std::unique_ptr<Partition> bloat(
    const tuple<std::uint64_t>& low_offsets,
    const tuple<std::uint64_t>& high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::unique_ptr<Partition> clone() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const tuple<std::uint64_t>& tile_shape() const;
  [[nodiscard]] const tuple<std::uint64_t>& color_shape() const override;
  [[nodiscard]] const tuple<std::int64_t>& offsets() const;
  [[nodiscard]] bool has_color(const tuple<std::uint64_t>& color) const;

  [[nodiscard]] tuple<std::uint64_t> get_child_extents(const tuple<std::uint64_t>& extents,
                                                       const tuple<std::uint64_t>& color) const;
  [[nodiscard]] tuple<std::uint64_t> get_child_offsets(const tuple<std::uint64_t>& color) const;

  [[nodiscard]] std::size_t hash() const;

 private:
  bool overlapped_{};
  tuple<std::uint64_t> tile_shape_{};
  tuple<std::uint64_t> color_shape_{};
  tuple<std::int64_t> offsets_{};
  tuple<std::uint64_t> strides_{};
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

  [[nodiscard]] std::unique_ptr<Partition> scale(
    const tuple<std::uint64_t>& factors) const override;
  [[nodiscard]] std::unique_ptr<Partition> bloat(
    const tuple<std::uint64_t>& low_offsets,
    const tuple<std::uint64_t>& high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::unique_ptr<Partition> clone() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const tuple<std::uint64_t>& color_shape() const override;

 private:
  std::unique_ptr<Legion::FutureMap> weights_{nullptr};
  Domain color_domain_{};
  tuple<std::uint64_t> color_shape_{};
};

class Image : public Partition {
 public:
  Image(InternalSharedPtr<detail::LogicalStore> func,
        InternalSharedPtr<Partition> func_partition,
        mapping::detail::Machine machine);

  bool operator==(const Image& other) const;
  bool operator<(const Image& other) const;

  [[nodiscard]] Kind kind() const override;

  [[nodiscard]] bool is_complete_for(const detail::Storage* storage) const override;
  [[nodiscard]] bool is_disjoint_for(const Domain& launch_domain) const override;
  [[nodiscard]] bool satisfies_restrictions(const Restrictions& restrictions) const override;
  [[nodiscard]] bool is_convertible() const override;

  [[nodiscard]] std::unique_ptr<Partition> scale(
    const tuple<std::uint64_t>& factors) const override;
  [[nodiscard]] std::unique_ptr<Partition> bloat(
    const tuple<std::uint64_t>& low_offsets,
    const tuple<std::uint64_t>& high_offsets) const override;

  [[nodiscard]] Legion::LogicalPartition construct(Legion::LogicalRegion region,
                                                   bool complete) const override;

  [[nodiscard]] bool has_launch_domain() const override;
  [[nodiscard]] Domain launch_domain() const override;

  [[nodiscard]] std::unique_ptr<Partition> clone() const override;

  [[nodiscard]] std::string to_string() const override;

  [[nodiscard]] const tuple<std::uint64_t>& color_shape() const override;

 private:
  InternalSharedPtr<detail::LogicalStore> func_;
  InternalSharedPtr<Partition> func_partition_{};
  mapping::detail::Machine machine_{};
};

[[nodiscard]] std::unique_ptr<NoPartition> create_no_partition();

[[nodiscard]] std::unique_ptr<Tiling> create_tiling(tuple<std::uint64_t>&& tile_shape,
                                                    tuple<std::uint64_t>&& color_shape,
                                                    tuple<std::int64_t>&& offsets = {});

[[nodiscard]] std::unique_ptr<Tiling> create_tiling(tuple<std::uint64_t>&& tile_shape,
                                                    tuple<std::uint64_t>&& color_shape,
                                                    tuple<std::int64_t>&& offsets,
                                                    tuple<std::uint64_t>&& strides);

[[nodiscard]] std::unique_ptr<Weighted> create_weighted(const Legion::FutureMap& weights,
                                                        const Domain& color_domain);

[[nodiscard]] std::unique_ptr<Image> create_image(InternalSharedPtr<detail::LogicalStore> func,
                                                  InternalSharedPtr<Partition> func_partition,
                                                  mapping::detail::Machine machine);

std::ostream& operator<<(std::ostream& out, const Partition& partition);

}  // namespace legate

#include "core/partitioning/partition.inl"
