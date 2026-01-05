/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/partition/image.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/transform/non_invertible_transformation.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/partitioning/constraint.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>
#include <legate/utilities/typedefs.h>

#include <fmt/format.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace legate::detail {

Image::Image(InternalSharedPtr<detail::LogicalStore> func,
             InternalSharedPtr<Partition> func_partition,
             mapping::detail::Machine machine,
             ImageComputationHint hint)
  : func_{std::move(func)},
    func_partition_{std::move(func_partition)},
    machine_{std::move(machine)},
    hint_{hint}
{
}

bool Image::operator==(const Image& other) const
{
  return func_->id() == other.func_->id() && func_partition_ == other.func_partition_ &&
         hint_ == other.hint_;
}

bool Image::is_disjoint_for(const Domain& launch_domain) const
{
  // Disjointedness check for image partitions is expensive, so we give a sound answer;
  return !launch_domain.is_valid();
}

InternalSharedPtr<Partition> Image::scale(Span<const std::uint64_t> /*factors*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
}

InternalSharedPtr<Partition> Image::bloat(Span<const std::uint64_t> /*low_offsts*/,
                                          Span<const std::uint64_t> /*high_offsets*/) const
{
  throw TracedException<std::runtime_error>{"Not implemented"};
}

Legion::LogicalPartition Image::construct(Legion::LogicalRegion region, bool /*complete*/) const
{
  if (!has_launch_domain()) {
    return Legion::LogicalPartition::NO_PART;
  }

  auto&& func_rf      = func_->get_region_field();
  auto&& func_region  = func_rf->region();
  auto func_partition = func_partition_->construct(
    func_region, func_partition_->is_complete_for(*func_->get_storage()));

  auto&& runtime  = detail::Runtime::get_runtime();
  auto&& part_mgr = runtime.partition_manager();

  auto target          = region.get_index_space();
  const auto field_id  = func_rf->field_id();
  auto index_partition = part_mgr.find_image_partition(target, func_partition, field_id, hint_);

  if (Legion::IndexPartition::NO_PART == index_partition) {
    auto construct_image_partition = [&] {
      switch (hint_) {
        case ImageComputationHint::NO_HINT: {
          const bool is_range = func_->type()->code == Type::Code::STRUCT;
          auto color_space    = runtime.find_or_create_index_space(color_shape());
          return runtime.create_image_partition(
            target, color_space, func_region, func_partition, field_id, is_range, machine_);
        }
        case ImageComputationHint::MIN_MAX: {
          return runtime.create_approximate_image_partition(
            func_, func_partition_, target, /*sorted=*/false);
        }
        case ImageComputationHint::FIRST_LAST: {
          return runtime.create_approximate_image_partition(
            func_, func_partition_, target, /*sorted=*/true);
        }
      }
      LEGATE_ABORT("Unhandled image hint ", to_underlying(hint_));
    };

    index_partition = construct_image_partition();
    part_mgr.record_image_partition(target, func_partition, field_id, hint_, index_partition);
    func_rf->add_invalidation_callback([target, func_partition, field_id, hint = hint_]() noexcept {
      detail::Runtime::get_runtime().partition_manager().invalidate_image_partition(
        target, func_partition, field_id, hint);
    });
  }

  return runtime.create_logical_partition(region, index_partition);
}

bool Image::has_launch_domain() const { return func_partition_->has_launch_domain(); }

Domain Image::launch_domain() const { return func_partition_->launch_domain(); }

std::string Image::to_string() const
{
  return fmt::format("Image(func: {}, partition: {}, hint: {})",
                     func_->to_string(),
                     func_partition_->to_string(),
                     hint_);
}

bool Image::has_color_shape() const { return func_partition_->has_color_shape(); }

Span<const std::uint64_t> Image::color_shape() const { return func_partition_->color_shape(); }

InternalSharedPtr<Partition> Image::convert(
  const InternalSharedPtr<Partition>& self,
  const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
}

InternalSharedPtr<Partition> Image::invert(const InternalSharedPtr<Partition>& self,
                                           const InternalSharedPtr<TransformStack>& transform) const
{
  if (transform->identity()) {
    return self;
  }
  throw TracedException<NonInvertibleTransformation>{};
}

InternalSharedPtr<Image> create_image(InternalSharedPtr<detail::LogicalStore> func,
                                      InternalSharedPtr<Partition> func_partition,
                                      mapping::detail::Machine machine,
                                      ImageComputationHint hint)
{
  return make_internal_shared<Image>(
    std::move(func), std::move(func_partition), std::move(machine), hint);
}

}  // namespace legate::detail
