/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/proxy.h>
#include <legate/utilities/shared_ptr.h>

#include <optional>
#include <variant>

namespace legate::mapping::detail {

class ProxyStoreMapping;

}  // namespace legate::mapping::detail

namespace legate::mapping {

enum class StoreTarget : std::uint8_t;

class ProxyInstanceMappingPolicy;

/**
 * @brief Model a store mapping descriptor that can be applied to one or more arguments of a task.
 */
class LEGATE_EXPORT ProxyStoreMapping {
 public:
  ProxyStoreMapping()                                        = delete;
  ProxyStoreMapping(const ProxyStoreMapping&)                = default;
  ProxyStoreMapping& operator=(const ProxyStoreMapping&)     = default;
  ProxyStoreMapping(ProxyStoreMapping&&) noexcept            = default;
  ProxyStoreMapping& operator=(ProxyStoreMapping&&) noexcept = default;
  ~ProxyStoreMapping();

  explicit ProxyStoreMapping(InternalSharedPtr<detail::ProxyStoreMapping> impl);

  /**
   * @brief Construct a store mapping.
   *
   * Builds an instance mapping policy with the provided arguments.
   *
   * @param store Store argument(s) to apply the mapping policy to.
   * @param target Specific store target or `std::nullopt` if the target should be chosen at
   * mapping-time. See `ProxyInstanceMappingPolicy` for more information.
   * @param exact Whether the mapping must be exact.
   */
  ProxyStoreMapping(std::variant<ProxyArrayArgument,
                                 ProxyInputArguments,
                                 ProxyOutputArguments,
                                 ProxyReductionArguments> store,
                    std::optional<StoreTarget> target,
                    bool exact = false);

  /**
   * @brief Construct a store mapping.
   *
   * @param store Store argument(s) to apply the mapping policy to.
   * @param policy Instance mapping policy.
   */
  ProxyStoreMapping(std::variant<ProxyArrayArgument,
                                 ProxyInputArguments,
                                 ProxyOutputArguments,
                                 ProxyReductionArguments> store,
                    ProxyInstanceMappingPolicy&& policy);

  /**
   * @return Associated instance mapping policy.
   */
  [[nodiscard]] const ProxyInstanceMappingPolicy& policy() const;

  [[nodiscard]] const SharedPtr<detail::ProxyStoreMapping>& impl() const;

  friend bool operator==(const ProxyStoreMapping& lhs, const ProxyStoreMapping& rhs);
  friend bool operator!=(const ProxyStoreMapping& lhs, const ProxyStoreMapping& rhs);

 private:
  SharedPtr<detail::ProxyStoreMapping> impl_{};
};

}  // namespace legate::mapping

#include <legate/mapping/proxy_store_mapping.inl>
