/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/detail/proxy_store_mapping.h>
#include <legate/mapping/mapping.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/span.h>

#include <vector>

namespace legate::mapping::detail {

class Task;

}  // namespace legate::mapping::detail

namespace legate::detail {

/**
 * @brief Holds a collection of proxy store mappings and can apply them to a task to produce
 * concrete store mappings.
 */
class StoreMappingSignature {
 public:
  /**
   * @brief Construct a store mapping signature.
   *
   * @param store_mappings Proxy store mappings that define the signature.
   */
  explicit StoreMappingSignature(
    SmallVector<InternalSharedPtr<mapping::detail::ProxyStoreMapping>> store_mappings);

  /**
   * @brief Apply the signature to a task instance.
   *
   * @param task Task to which the store mappings are applied.
   * @param options Store target options for the task invocation.
   *
   * @return Concrete store mappings produced for the task.
   */
  [[nodiscard]] std::vector<mapping::StoreMapping> apply(
    const mapping::detail::Task& task, Span<const mapping::StoreTarget> options) const;

  /**
   * @brief Access the proxy store mappings.
   *
   * @return Span over the proxy store mappings in this signature.
   */
  [[nodiscard]] Span<const InternalSharedPtr<mapping::detail::ProxyStoreMapping>> store_mappings()
    const;

  friend bool operator==(const StoreMappingSignature& lhs, const StoreMappingSignature& rhs);
  friend bool operator!=(const StoreMappingSignature& lhs, const StoreMappingSignature& rhs);

 private:
  SmallVector<InternalSharedPtr<mapping::detail::ProxyStoreMapping>> store_mappings_{};
};

}  // namespace legate::detail

#include <legate/task/detail/store_mapping_signature.inl>
