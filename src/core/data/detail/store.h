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

#include "core/data/buffer.h"
#include "core/task/detail/return.h"
#include "core/type/detail/type_info.h"

namespace legate {
class Store;
}  // namespace legate
namespace legate::detail {

class BaseArray;
class TransformStack;

class RegionField {
 public:
  RegionField() {}
  RegionField(int32_t dim, const Legion::PhysicalRegion& pr, Legion::FieldID fid);

 public:
  RegionField(RegionField&& other) noexcept;
  RegionField& operator=(RegionField&& other) noexcept;

 private:
  RegionField(const RegionField& other)            = delete;
  RegionField& operator=(const RegionField& other) = delete;

 public:
  bool valid() const;

 public:
  int32_t dim() const { return dim_; }

 public:
  Domain domain() const;

 public:
  void unmap();

 public:
  bool is_readable() const { return readable_; }
  bool is_writable() const { return writable_; }
  bool is_reducible() const { return reducible_; }

 public:
  Legion::PhysicalRegion get_physical_region() const { return pr_; }
  Legion::FieldID get_field_id() const { return fid_; }

 private:
  int32_t dim_{-1};
  Legion::PhysicalRegion pr_{};
  Legion::FieldID fid_{-1U};

 private:
  bool readable_{false};
  bool writable_{false};
  bool reducible_{false};
};

class UnboundRegionField {
 public:
  UnboundRegionField() {}
  UnboundRegionField(const Legion::OutputRegion& out, Legion::FieldID fid);

 public:
  UnboundRegionField(UnboundRegionField&& other) noexcept;
  UnboundRegionField& operator=(UnboundRegionField&& other) noexcept;

 private:
  UnboundRegionField(const UnboundRegionField& other)            = delete;
  UnboundRegionField& operator=(const UnboundRegionField& other) = delete;

 public:
  bool bound() const { return bound_; }

 public:
  void bind_empty_data(int32_t dim);

 public:
  ReturnValue pack_weight() const;

 public:
  void set_bound(bool bound) { bound_ = bound; }
  void update_num_elements(size_t num_elements);

 public:
  Legion::OutputRegion get_output_region() const { return out_; }
  Legion::FieldID get_field_id() const { return fid_; }

 private:
  bool bound_{false};
  Legion::UntypedDeferredValue num_elements_;
  Legion::OutputRegion out_{};
  Legion::FieldID fid_{-1U};
};

class FutureWrapper {
 public:
  FutureWrapper() {}
  FutureWrapper(bool read_only,
                uint32_t field_size,
                Domain domain,
                Legion::Future future,
                bool initialize = false);

 public:
  FutureWrapper(const FutureWrapper& other) noexcept;
  FutureWrapper& operator=(const FutureWrapper& other) noexcept;

 public:
  int32_t dim() const { return domain_.dim; }

 public:
  Domain domain() const;

 public:
  void initialize_with_identity(int32_t redop_id);

 public:
  ReturnValue pack() const;

 public:
  bool is_read_only() const { return read_only_; }
  Legion::Future get_future() const { return future_; }
  Legion::UntypedDeferredValue get_buffer() const { return buffer_; }

 private:
  bool read_only_{true};
  uint32_t field_size_{0};
  Domain domain_{};
  Legion::Future future_{};
  Legion::UntypedDeferredValue buffer_{};
};

class Store {
 public:
  Store() {}
  Store(int32_t dim,
        std::shared_ptr<Type> type,
        int32_t redop_id,
        FutureWrapper future,
        std::shared_ptr<detail::TransformStack>&& transform = nullptr);
  Store(int32_t dim,
        std::shared_ptr<Type> type,
        int32_t redop_id,
        RegionField&& region_field,
        std::shared_ptr<detail::TransformStack>&& transform = nullptr);
  Store(int32_t dim,
        std::shared_ptr<Type> type,
        UnboundRegionField&& unbound_field,
        std::shared_ptr<detail::TransformStack>&& transform = nullptr);
  Store(int32_t dim,
        std::shared_ptr<Type> type,
        int32_t redop_id,
        FutureWrapper future,
        const std::shared_ptr<detail::TransformStack>& transform);
  Store(int32_t dim,
        std::shared_ptr<Type> type,
        int32_t redop_id,
        RegionField&& region_field,
        const std::shared_ptr<detail::TransformStack>& transform);

 public:
  Store(Store&& other) noexcept;
  Store& operator=(Store&& other) noexcept;

 private:
  Store(const Store& other)            = delete;
  Store& operator=(const Store& other) = delete;

 public:
  bool valid() const;
  bool transformed() const;

 public:
  int32_t dim() const { return dim_; }
  std::shared_ptr<Type> type() const { return type_; }

 public:
  Domain domain() const;

 public:
  void unmap();

 public:
  bool is_readable() const { return readable_; }
  bool is_writable() const { return writable_; }
  bool is_reducible() const { return reducible_; }

 public:
  void bind_empty_data();

 public:
  bool is_future() const;
  bool is_unbound_store() const;
  ReturnValue pack() const { return future_.pack(); }
  ReturnValue pack_weight() const { return unbound_field_.pack_weight(); }

 private:
  friend class legate::Store;
  friend class legate::detail::BaseArray;
  void check_accessor_dimension(const int32_t dim) const;
  void check_buffer_dimension(const int32_t dim) const;
  void check_shape_dimension(const int32_t dim) const;
  void check_valid_binding(bool bind_buffer) const;
  Legion::DomainAffineTransform get_inverse_transform() const;

 private:
  void get_region_field(Legion::PhysicalRegion& pr, Legion::FieldID& fid) const;
  int32_t get_redop_id() const { return redop_id_; }

 private:
  bool is_read_only_future() const;
  Legion::Future get_future() const;
  Legion::UntypedDeferredValue get_buffer() const;

 private:
  void get_output_field(Legion::OutputRegion& out, Legion::FieldID& fid);
  void update_num_elements(size_t num_elements);

 private:
  bool is_future_{false};
  bool is_unbound_store_{false};
  int32_t dim_{-1};
  std::shared_ptr<Type> type_{};
  int32_t redop_id_{-1};

 private:
  FutureWrapper future_;
  RegionField region_field_;
  UnboundRegionField unbound_field_;

 private:
  std::shared_ptr<detail::TransformStack> transform_{nullptr};

 private:
  bool readable_{false};
  bool writable_{false};
  bool reducible_{false};
};

}  // namespace legate::detail
