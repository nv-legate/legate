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

#include <map>
#include <unordered_map>

#include "core/mapping/detail/machine.h"
#include "core/utilities/typedefs.h"

namespace legate::detail {

// TODO: We need to expose this eventually so client libraries can register custom communicators
class CommunicatorFactory {
 protected:
  CommunicatorFactory();

 public:
  virtual ~CommunicatorFactory();

 public:
  Legion::FutureMap find_or_create(const mapping::TaskTarget& target,
                                   const mapping::ProcessorRange& range,
                                   const Domain& launch_domain);
  void destroy();

 protected:
  Legion::FutureMap find_or_create(const mapping::TaskTarget& target,
                                   const mapping::ProcessorRange& range,
                                   uint32_t num_tasks);
  Legion::FutureMap transform(const Legion::FutureMap& communicator, const Domain& launch_domain);

 public:
  virtual bool needs_barrier() const                                 = 0;
  virtual bool is_supported_target(mapping::TaskTarget target) const = 0;

 protected:
  virtual Legion::FutureMap initialize(const mapping::detail::Machine& machine,
                                       uint32_t num_tasks)     = 0;
  virtual void finalize(const mapping::detail::Machine& machine,
                        uint32_t num_tasks,
                        const Legion::FutureMap& communicator) = 0;

 private:
  template <class Desc>
  struct CacheKey {
    Desc desc;
    mapping::TaskTarget target;
    mapping::ProcessorRange range;
    mapping::detail::Machine get_machine() const
    {
      return mapping::detail::Machine({{target, range}});
    }
    bool operator==(const CacheKey& other) const
    {
      return desc == other.desc && target == other.target && range == other.range;
    }
    bool operator<(const CacheKey& other) const
    {
      if (desc < other.desc)
        return true;
      else if (other.desc < desc)
        return false;
      if (target < other.target)
        return true;
      else if (target > other.target)
        return false;
      return range < other.range;
    }
  };
  using CommKey  = CacheKey<uint32_t>;
  using AliasKey = CacheKey<Domain>;
  std::map<CommKey, Legion::FutureMap> communicators_;
  std::map<AliasKey, Legion::FutureMap> nd_aliases_;
};

class CommunicatorManager {
 public:
  CommunicatorFactory* find_factory(const std::string& name);
  void register_factory(const std::string& name, std::unique_ptr<CommunicatorFactory> factory);

 public:
  void destroy();

 private:
  std::unordered_map<std::string, std::unique_ptr<CommunicatorFactory>> factories_;
};

}  // namespace legate::detail
