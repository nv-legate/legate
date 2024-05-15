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

#include "core/data/inline_allocation.h"
#include "core/mapping/mapping.h"
#include "core/task/detail/return_value.h"
#include "core/utilities/typedefs.h"

#include <cstdint>

namespace legate::detail {

class FutureWrapper {
 public:
  FutureWrapper() = default;
  FutureWrapper(bool read_only,
                std::uint32_t field_size,
                const Domain& domain,
                Legion::Future future);

  [[nodiscard]] std::int32_t dim() const;
  [[nodiscard]] const Domain& domain() const;
  [[nodiscard]] bool valid() const;
  [[nodiscard]] std::uint32_t field_size() const;

  [[nodiscard]] InlineAllocation get_inline_allocation(const Domain& domain) const;
  [[nodiscard]] InlineAllocation get_inline_allocation() const;
  [[nodiscard]] mapping::StoreTarget target() const;

  void initialize_with_identity(std::int32_t redop_id);

  [[nodiscard]] ReturnValue pack() const;

  [[nodiscard]] bool is_read_only() const;
  [[nodiscard]] const Legion::Future& get_future() const;
  [[nodiscard]] const Legion::UntypedDeferredValue& get_buffer() const;

 private:
  bool read_only_{true};
  std::uint32_t field_size_{};
  Domain domain_{};
  Legion::Future future_{};
  Legion::UntypedDeferredValue buffer_{};
};

}  // namespace legate::detail

#include "core/data/detail/future_wrapper.inl"
