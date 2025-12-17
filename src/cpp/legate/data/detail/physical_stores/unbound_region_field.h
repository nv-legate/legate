/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/return_value.h>

#include <legion/api/types.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {

class UnboundRegionField {
 public:
  UnboundRegionField() = default;
  UnboundRegionField(const Legion::OutputRegion& out, Legion::FieldID fid, bool partitioned);

  UnboundRegionField(UnboundRegionField&& other) noexcept;
  UnboundRegionField& operator=(UnboundRegionField&& other) noexcept;

  UnboundRegionField(const UnboundRegionField& other)            = delete;
  UnboundRegionField& operator=(const UnboundRegionField& other) = delete;

  [[nodiscard]] bool is_partitioned() const;
  [[nodiscard]] bool bound() const;

  void bind_empty_data(std::int32_t dim);

  void set_bound(bool bound);

  [[nodiscard]] const Legion::OutputRegion& get_output_region() const;
  [[nodiscard]] Legion::FieldID get_field_id() const;

 private:
  bool bound_{};
  bool partitioned_{};
  Legion::UntypedDeferredValue num_elements_{};
  Legion::OutputRegion out_{};
  Legion::FieldID fid_{-1U};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_stores/unbound_region_field.inl>
