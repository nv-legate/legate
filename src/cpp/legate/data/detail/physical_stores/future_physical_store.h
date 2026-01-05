/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/future_wrapper.h>
#include <legate/data/detail/physical_store.h>
#include <legate/data/detail/transform/transform_stack.h>
#include <legate/data/inline_allocation.h>
#include <legate/task/detail/return_value.h>
#include <legate/type/detail/types.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

}

namespace legate::detail {

class FuturePhysicalStore final : public PhysicalStore {
 public:
  FuturePhysicalStore(std::int32_t dim,
                      InternalSharedPtr<Type> type,
                      GlobalRedopID redop_id,
                      FutureWrapper future,
                      InternalSharedPtr<detail::TransformStack> transform = nullptr);

  [[nodiscard]] bool valid() const override;
  [[nodiscard]] Domain domain() const override;
  [[nodiscard]] InlineAllocation get_inline_allocation() const override;
  [[nodiscard]] mapping::StoreTarget target() const override;
  [[nodiscard]] bool is_partitioned() const override;

  // Future specific API
  [[nodiscard]] const Legion::Future& get_future() const;
  [[nodiscard]] const Legion::UntypedDeferredValue& get_buffer() const;
  [[nodiscard]] ReturnValue pack() const;

  /**
   * @brief Updates the future associated with this PhysicalStore.
   *
   * This method is used by handle_return_values to update scalar output futures
   * after task execution completes. Only valid for future-backed PhysicalStores.
   *
   * @param future New future to associate with this store
   */
  void set_future(Legion::Future future);

  [[nodiscard]] bool is_read_only_future() const;
  [[nodiscard]] std::size_t get_field_offset() const;
  [[nodiscard]] const void* get_untyped_pointer_from_future() const;

 private:
  FutureWrapper future_{};
};

}  // namespace legate::detail

#include <legate/data/detail/physical_stores/future_physical_store.inl>
