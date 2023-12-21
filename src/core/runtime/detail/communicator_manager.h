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

#include "core/mapping/detail/machine.h"
#include "core/utilities/detail/hash.h"
#include "core/utilities/hash.h"
#include "core/utilities/typedefs.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace legate::detail {

// TODO: We need to expose this eventually so client libraries can register custom communicators
class CommunicatorFactory {
 public:
  CommunicatorFactory()                                               = default;
  CommunicatorFactory(const CommunicatorFactory&) noexcept            = default;
  CommunicatorFactory& operator=(const CommunicatorFactory&) noexcept = default;
  CommunicatorFactory(CommunicatorFactory&&) noexcept                 = default;
  CommunicatorFactory& operator=(CommunicatorFactory&&) noexcept      = default;

  virtual ~CommunicatorFactory() = default;

  [[nodiscard]] Legion::FutureMap find_or_create(const mapping::TaskTarget& target,
                                                 const mapping::ProcessorRange& range,
                                                 const Domain& launch_domain);
  void destroy();

 protected:
  [[nodiscard]] Legion::FutureMap find_or_create(const mapping::TaskTarget& target,
                                                 const mapping::ProcessorRange& range,
                                                 uint32_t num_tasks);
  [[nodiscard]] Legion::FutureMap transform(const Legion::FutureMap& communicator,
                                            const Domain& launch_domain);

 public:
  [[nodiscard]] virtual bool needs_barrier() const                                 = 0;
  [[nodiscard]] virtual bool is_supported_target(mapping::TaskTarget target) const = 0;

 protected:
  [[nodiscard]] virtual Legion::FutureMap initialize(const mapping::detail::Machine& machine,
                                                     uint32_t num_tasks) = 0;
  virtual void finalize(const mapping::detail::Machine& machine,
                        uint32_t num_tasks,
                        const Legion::FutureMap& communicator)           = 0;

 private:
  template <class Desc>
  struct CacheKey {
    Desc desc;
    mapping::TaskTarget target;
    mapping::ProcessorRange range;

    [[nodiscard]] mapping::detail::Machine get_machine() const;
    bool operator==(const CacheKey& other) const;
    [[nodiscard]] size_t hash() const noexcept;
  };
  using CommKey  = CacheKey<uint32_t>;
  using AliasKey = CacheKey<Domain>;

  std::unordered_map<CommKey, Legion::FutureMap, hasher<CommKey>> communicators_{};
  std::unordered_map<AliasKey, Legion::FutureMap, hasher<AliasKey>> nd_aliases_{};
};

class CommunicatorManager {
 public:
  [[nodiscard]] CommunicatorFactory* find_factory(const std::string& name);
  void register_factory(const std::string& name, std::unique_ptr<CommunicatorFactory> factory);

  void destroy();

 private:
  std::unordered_map<std::string, std::unique_ptr<CommunicatorFactory>> factories_{};
};

}  // namespace legate::detail

#include "core/runtime/detail/communicator_manager.inl"
