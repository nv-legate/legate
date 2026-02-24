/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/mapping/mapping.h>
#include <legate/mapping/proxy_instance_mapping_policy.h>
#include <legate/partitioning/proxy.h>
#include <legate/utilities/detail/small_vector.h>

#include <variant>
#include <vector>

namespace legate::detail {

class TaskBase;

}  // namespace legate::detail

namespace legate::mapping {

class StoreMapping;

}  // namespace legate::mapping

namespace legate::mapping::detail {

class Task;

/**
 * @brief Binds one or more proxy stores to an instance mapping policy.
 *
 * Encapsulates a set of proxy store arguments (array, input, output, or reduction) together
 * with an InstanceMappingPolicy, and applies that policy to produce concrete store mappings
 * for a task.
 */
class ProxyStoreMapping {
 public:
  /**
   * @brief Construct a proxy store mapping.
   *
   * @param store  Proxy store argument(s) to be mapped.
   * @param policy Proxy instance mapping policy to use to synthesize the instance mapping policy.
   */
  ProxyStoreMapping(std::variant<ProxyArrayArgument,
                                 ProxyInputArguments,
                                 ProxyOutputArguments,
                                 ProxyReductionArguments> store,
                    ProxyInstanceMappingPolicy policy);

  /**
   * @brief Apply the mapping policy to a task.
   *
   * Populates `store_mappings` with concrete mappings derived from the stored proxy arguments
   * and mapping policy.
   *
   * @param task           Task to which the mappings are to be applied.
   * @param options        Task specific store target options.
   * @param store_mappings Output container for generated store mappings.
   */
  void apply_legion(const Task& task,
                    Span<const StoreTarget> options,
                    std::vector<mapping::StoreMapping>* store_mappings) const;

  /**
   * @brief Apply the mapping policy to task being inline-executed.
   *
   * This routine assumes that the policy vectors are already presized to the number of input,
   * output, and reduction arguments to a task.
   *
   * @param task The task to apply the policies to.
   * @param options Task specific store target options.
   * @param input_policies The destination specific policies for the input arguments.
   * @param output_policies The destination specific policies for the output arguments.
   * @param reduction_policies The destination specific policies for the reduction arguments.
   */
  void apply_inline(const legate::detail::TaskBase& task,
                    Span<const StoreTarget> options,
                    legate::detail::SmallVector<InstanceMappingPolicy>* input_policies,
                    legate::detail::SmallVector<InstanceMappingPolicy>* output_policies,
                    legate::detail::SmallVector<InstanceMappingPolicy>* reduction_policies) const;

  /**
   * @return The proxy store arguments.
   */
  [[nodiscard]] const std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>&
    stores() const;

  /**
   * @return Mapping policy associated with this object.
   */
  [[nodiscard]] const ProxyInstanceMappingPolicy& policy() const;

  /** @brief Equality comparison */
  friend bool operator==(const ProxyStoreMapping& lhs, const ProxyStoreMapping& rhs);

  /** @brief Inequality comparison */
  friend bool operator!=(const ProxyStoreMapping& lhs, const ProxyStoreMapping& rhs);

 private:
  std::
    variant<ProxyArrayArgument, ProxyInputArguments, ProxyOutputArguments, ProxyReductionArguments>
      stores_{};
  ProxyInstanceMappingPolicy policy_{};
};

}  // namespace legate::mapping::detail

#include <legate/mapping/detail/proxy_store_mapping.inl>
