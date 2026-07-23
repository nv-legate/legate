/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/opaque.h>

#include <legate/data/detail/transform/transform_stack.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/formatters.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/detail/tuple.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace legate::detail {

bool Opaque::operator==(const Opaque& other) const
{
  return filled_ == other.filled_ && color_shape_ == other.color_shape_ &&
         std::equal(colors_.begin(), colors_.end(), other.colors_.begin()) &&
         std::equal(domains_.begin(), domains_.end(), other.domains_.begin());
}

InternalSharedPtr<Partition> Opaque::scale(Span<const std::uint64_t> /*factors*/) const
{
  throw TracedException<std::runtime_error>{"Scaling is not supported for opaque partitions"};
}

InternalSharedPtr<Partition> Opaque::bloat(Span<const std::uint64_t> /*low_offsets*/,
                                           Span<const std::uint64_t> /*high_offsets*/) const
{
  throw TracedException<std::runtime_error>{"Bloating is not supported for opaque partitions"};
}

Rect Opaque::launch_domain() const { return Rect::from_extents(color_shape_); }

std::optional<std::uint64_t> Opaque::device_of_(const Point& color) const
{
  for (auto&& [device_id, my_color] : enumerate(colors_)) {
    if (my_color == color) {
      return static_cast<std::uint64_t>(device_id);
    }
  }
  return std::nullopt;
}

Point Opaque::compute_color(std::uint64_t device_id) const
{
  if (device_id >= colors_.size()) {
    return {};
  }
  return colors_[device_id];
}

std::vector<std::uint64_t> Opaque::compute_device_ids(const Point& color) const
{
  if (const auto device_id = device_of_(color); device_id.has_value()) {
    return {*device_id};
  }
  return {};
}

Rect Opaque::shard(const Rect& domain, const Point& color) const
{
  if (const auto device_id = device_of_(color); device_id.has_value()) {
    return domains_[*device_id];
  }
  // An unknown color owns no piece.
  return domain;
}

bool Opaque::subsumes(const Rect&, const InternalSharedPtr<Partition>& other) const
{
  if (other.get() == this) {
    return true;
  }
  const auto* other_opaque = dynamic_cast<const Opaque*>(other.get());

  return other_opaque != nullptr && *this == *other_opaque;
}

std::string Opaque::to_string() const
{
  // Stream the per-piece color->domain pairs directly; interleaving the `->` separator with the
  // running device index reads more clearly as a loop than as a single fmt call.
  std::stringstream ss;

  ss << fmt::format("Opaque(colors: {}, pieces: [", color_shape_);
  for (auto&& [device_id, dom] : enumerate(domains_)) {
    ss << (device_id == 0 ? "" : ", ") << colors_[device_id] << "->" << dom;
  }
  ss << "])";
  return std::move(ss).str();
}

InternalSharedPtr<Partition> Opaque::convert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<std::runtime_error>{
    "An opaque partition cannot be converted through a non-identity transformation"};
}

InternalSharedPtr<Partition> Opaque::invert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<std::runtime_error>{
    "An opaque partition cannot be inverted through a non-identity transformation"};
}

void Opaque::fill(SmallVector<Point> colors, SmallVector<Rect> domains)
{
  LEGATE_ASSERT(colors.size() == domains.size());
  colors_  = std::move(colors);
  domains_ = std::move(domains);
  filled_  = true;
}

// ==========================================================================================

InternalSharedPtr<Opaque> create_opaque(Point color_shape)
{
  return make_internal_shared<Opaque>(std::move(color_shape));
}

}  // namespace legate::detail
