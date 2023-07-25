/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <functional>
#include <memory>
#include "core/data/scalar.h"
#include "core/mapping/store.h"

/** @defgroup mapping Mapping API
 */

/**
 * @file
 * @brief Legate Mapping API
 */

namespace legate::mapping {

namespace detail {
class BaseMapper;
class DimOrdering;
class StoreMapping;
}  // namespace detail

class Task;

// NOTE: codes are chosen to reflect the precedence between the processor kinds in choosing target
// processors for tasks.

/**
 * @ingroup mapping
 * @brief An enum class for task targets
 *
 * The enumerators of `TaskTarget` are ordered by their precedence; i.e., `GPU`, if available, is
 * chosen over `OMP` or `CPU, `OMP`, if available, is chosen over `CPU`.
 */
enum class TaskTarget : int32_t {
  /**
   * @brief Indicates the task be mapped to a GPU
   */
  GPU = 1,
  /**
   * @brief Indicates the task be mapped to an OpenMP processor
   */
  OMP = 2,
  /**
   * @brief Indicates the task be mapped to a CPU
   */
  CPU = 3,
};

std::ostream& operator<<(std::ostream& stream, const TaskTarget& target);

/**
 * @ingroup mapping
 * @brief An enum class for store targets
 */
enum class StoreTarget : int32_t {
  /**
   * @brief Indicates the store be mapped to the system memory (host memory)
   */
  SYSMEM = 1,
  /**
   * @brief Indicates the store be mapped to the GPU framebuffer
   */
  FBMEM = 2,
  /**
   * @brief Indicates the store be mapped to the pinned memory for zero-copy GPU accesses
   */
  ZCMEM = 3,
  /**
   * @brief Indicates the store be mapped to the host memory closest to the target CPU
   */
  SOCKETMEM = 4,
};

std::ostream& operator<<(std::ostream& stream, const StoreTarget& target);

/**
 * @ingroup mapping
 * @brief An enum class for instance allocation policies
 */
enum class AllocPolicy : int32_t {
  /**
   * @brief Indicates the store can reuse an existing instance
   */
  MAY_ALLOC = 1,
  /**
   * @brief Indicates the store must be mapped to a fresh instance
   */
  MUST_ALLOC = 2,
};

/**
 * @ingroup mapping
 * @brief An enum class for instant layouts
 */
enum class InstLayout : int32_t {
  /**
   * @brief Indicates the store must be mapped to an SOA instance
   */
  SOA = 1,
  /**
   * @brief Indicates the store must be mapped to an AOS instance. No different than `SOA` in a
   * store mapping for a single store
   */
  AOS = 2,
};

/**
 * @ingroup mapping
 * @brief A descriptor for dimension ordering
 */
struct DimOrdering {
 public:
  /**
   * @brief An enum class for kinds of dimension ordering
   */
  enum class Kind : int32_t {
    /**
     * @brief Indicates the instance have C layout (i.e., the last dimension is the leading
     * dimension in the instance)
     */
    C = 1,
    /**
     * @brief Indicates the instance have Fortran layout (i.e., the first dimension is the leading
     * dimension instance)
     */
    FORTRAN = 2,
    /**
     * @brief Indicates the order of dimensions of the instance is manually specified
     */
    CUSTOM = 3,
  };

 public:
  /**
   * @brief Creates a C ordering object
   *
   * @return A `DimOrdering` object
   */
  static DimOrdering c_order();
  /**
   * @brief Creates a Fortran ordering object
   *
   * @return A `DimOrdering` object
   */
  static DimOrdering fortran_order();
  /**
   * @brief Creates a custom ordering object
   *
   * @param dims A vector that stores the order of dimensions.
   *
   * @return A `DimOrdering` object
   */
  static DimOrdering custom_order(const std::vector<int32_t>& dims);

 public:
  /**
   * @brief Sets the dimension ordering to C
   */
  void set_c_order();
  /**
   * @brief Sets the dimension ordering to Fortran
   */
  void set_fortran_order();
  /**
   * @brief Sets a custom dimension ordering
   *
   * @param dims A vector that stores the order of dimensions.
   */
  void set_custom_order(const std::vector<int32_t>& dims);

 public:
  /**
   * @brief Dimension ordering type
   */
  Kind kind() const;
  /**
   * @brief Dimension list. Used only when the `kind` is `CUSTOM`.
   */
  std::vector<int32_t> dimensions() const;

 public:
  bool operator==(const DimOrdering&) const;

 private:
  DimOrdering(std::shared_ptr<detail::DimOrdering> impl);

 public:
  const detail::DimOrdering* impl() const;

 public:
  DimOrdering();
  DimOrdering(const DimOrdering&);
  DimOrdering& operator=(const DimOrdering&);
  DimOrdering(DimOrdering&&);
  DimOrdering& operator=(DimOrdering&&);

 private:
  std::shared_ptr<detail::DimOrdering> impl_;
};

/**
 * @ingroup mapping
 * @brief A descriptor for instance mapping policy
 */
struct InstanceMappingPolicy {
 public:
  /**
   * @brief Target memory type for the instance
   */
  StoreTarget target{StoreTarget::SYSMEM};
  /**
   * @brief Allocation policy
   */
  AllocPolicy allocation{AllocPolicy::MAY_ALLOC};
  /**
   * @brief Instance layout for the instance
   */
  InstLayout layout{InstLayout::SOA};
  /**
   * @brief Dimension ordering for the instance. C order by default.
   */
  DimOrdering ordering{};
  /**
   * @brief If true, the instance must be tight to the store(s); i.e., the instance
   * must not have any extra elements not included in the store(s).
   */
  bool exact{false};

 public:
  /**
   * @brief Changes the store target
   *
   * @param `target` A new store target
   *
   * @return This instance mapping policy
   */
  InstanceMappingPolicy& with_target(StoreTarget target) &;
  InstanceMappingPolicy&& with_target(StoreTarget target) const&;
  InstanceMappingPolicy&& with_target(StoreTarget target) &&;
  /**
   * @brief Changes the allocation policy
   *
   * @param `allocation` A new allocation policy
   *
   * @return This instance mapping policy
   */
  InstanceMappingPolicy& with_allocation_policy(AllocPolicy allocation) &;
  InstanceMappingPolicy&& with_allocation_policy(AllocPolicy allocation) const&;
  InstanceMappingPolicy&& with_allocation_policy(AllocPolicy allocation) &&;
  /**
   * @brief Changes the instance layout
   *
   * @param `target` A new instance layout
   *
   * @return This instance mapping policy
   */
  InstanceMappingPolicy& with_instance_layout(InstLayout layout) &;
  InstanceMappingPolicy&& with_instance_layout(InstLayout layout) const&;
  InstanceMappingPolicy&& with_instance_layout(InstLayout layout) &&;
  /**
   * @brief Changes the dimension ordering
   *
   * @param `target` A new dimension ordering
   *
   * @return This instance mapping policy
   */
  InstanceMappingPolicy& with_ordering(DimOrdering ordering) &;
  InstanceMappingPolicy&& with_ordering(DimOrdering ordering) const&;
  InstanceMappingPolicy&& with_ordering(DimOrdering ordering) &&;
  /**
   * @brief Changes the value of `exact`
   *
   * @param `target` A new value for the `exact` field
   *
   * @return This instance mapping policy
   */
  InstanceMappingPolicy& with_exact(bool exact) &;
  InstanceMappingPolicy&& with_exact(bool exact) const&;
  InstanceMappingPolicy&& with_exact(bool exact) &&;

 public:
  /**
   * @brief Changes the store target
   *
   * @param `target` A new store target
   */
  void set_target(StoreTarget target);
  /**
   * @brief Changes the allocation policy
   *
   * @param `allocation` A new allocation policy
   */
  void set_allocation_policy(AllocPolicy allocation);
  /**
   * @brief Changes the instance layout
   *
   * @param `target` A new instance layout
   */
  void set_instance_layout(InstLayout layout);
  /**
   * @brief Changes the dimension ordering
   *
   * @param `target` A new dimension ordering
   */
  void set_ordering(DimOrdering ordering);
  /**
   * @brief Changes the value of `exact`
   *
   * @param `target` A new value for the `exact` field
   */
  void set_exact(bool exact);

 public:
  /**
   * @brief Indicates whether this policy subsumes a given policy
   *
   * Policy `A` subsumes policy `B`, if every instance created under `B` satisfies `A` as well.
   *
   * @param other Policy to check the subsumption against
   *
   * @return true If this policy subsumes `other`
   * @return false Otherwise
   */
  bool subsumes(const InstanceMappingPolicy& other) const;

 public:
  InstanceMappingPolicy();
  ~InstanceMappingPolicy();

 public:
  InstanceMappingPolicy(const InstanceMappingPolicy&);
  InstanceMappingPolicy& operator=(const InstanceMappingPolicy&);
  InstanceMappingPolicy(InstanceMappingPolicy&&);
  InstanceMappingPolicy& operator=(InstanceMappingPolicy&&);

 public:
  bool operator==(const InstanceMappingPolicy&) const;
  bool operator!=(const InstanceMappingPolicy&) const;
};

/**
 * @ingroup mapping
 * @brief A mapping policy for stores
 */
struct StoreMapping {
 public:
  /**
   * @brief Creates a mapping policy for the given store following the default mapping poicy
   *
   * @param store Target store
   * @param target Kind of the memory to which the store should be mapped
   * @param exact Indicates whether the instance should be exact
   *
   * @return A store mapping
   */
  static StoreMapping default_mapping(Store store, StoreTarget target, bool exact = false);
  /**
   * @brief Creates a mapping policy for the given store using the instance mapping policy
   *
   * @param store Target store for the mapping policy
   * @param policy Instance mapping policy to apply
   *
   * @return A store mapping
   */
  static StoreMapping create(Store store, InstanceMappingPolicy&& policy);

  /**
   * @brief Creates a mapping policy for the given set of stores using the instance mapping policy
   *
   * @param store Target store for the mapping policy
   * @param policy Instance mapping policy to apply
   *
   * @return A store mapping
   */
  static StoreMapping create(const std::vector<Store>& stores, InstanceMappingPolicy&& policy);

 public:
  /**
   * @brief Returns the instance mapping policy of this `StoreMapping` object
   *
   * @return A reference to the `InstanceMappingPolicy` object
   */
  InstanceMappingPolicy& policy();
  /**
   * @brief Returns the instance mapping policy of this `StoreMapping` object
   *
   * @return A reference to the `InstanceMappingPolicy` object
   */
  const InstanceMappingPolicy& policy() const;

 public:
  /**
   * @brief Returns the store for which this `StoreMapping` object describes a mapping policy.
   *
   * If the policy is for multiple stores, the first store added to this policy will be returned;
   *
   * @return A `Store` object
   */
  Store store() const;
  /**
   * @brief Returns all the stores for which this `StoreMapping` object describes a mapping policy
   *
   * @return A vector of `Store` objects
   */
  std::vector<Store> stores() const;

 public:
  /**
   * @brief Adds a store to this `StoreMapping` object
   *
   * @param store Store to add
   */
  void add_store(Store store);

 private:
  StoreMapping(detail::StoreMapping* impl);

 public:
  const detail::StoreMapping* impl() const;

 private:
  StoreMapping(const StoreMapping&)            = delete;
  StoreMapping& operator=(const StoreMapping&) = delete;

 public:
  StoreMapping(StoreMapping&&);
  StoreMapping& operator=(StoreMapping&&);

 public:
  ~StoreMapping();

 private:
  friend class detail::BaseMapper;
  detail::StoreMapping* release();

 private:
  detail::StoreMapping* impl_{nullptr};
};

/**
 * @ingroup mapping
 * @brief An abstract class that defines machine query APIs
 */
class MachineQueryInterface {
 public:
  virtual ~MachineQueryInterface() {}
  /**
   * @brief Returns local CPUs
   *
   * @return A vector of processors
   */
  virtual const std::vector<Processor>& cpus() const = 0;
  /**
   * @brief Returns local GPUs
   *
   * @return A vector of processors
   */
  virtual const std::vector<Processor>& gpus() const = 0;
  /**
   * @brief Returns local OpenMP processors
   *
   * @return A vector of processors
   */
  virtual const std::vector<Processor>& omps() const = 0;
  /**
   * @brief Returns the total number of nodes
   *
   * @return Total number of nodes
   */
  virtual uint32_t total_nodes() const = 0;
};

/**
 * @ingroup mapping
 * @brief An abstract class that defines Legate mapping APIs
 *
 * The APIs give Legate libraries high-level control on task and store mappings
 */
class Mapper {
 public:
  virtual ~Mapper() {}
  /**
   * @brief Sets a machine query interface. This call gives the mapper a chance
   * to cache the machine query interface.
   *
   * @param machine Machine query interface
   */
  virtual void set_machine(const MachineQueryInterface* machine) = 0;
  /**
   * @brief Picks the target processor type for the task
   *
   * @param task Task to map
   * @param options Processor types for which the task has variants
   *
   * @return A target processor type
   */
  virtual TaskTarget task_target(const Task& task, const std::vector<TaskTarget>& options) = 0;
  /**
   * @brief Chooses mapping policies for the task's stores.
   *
   * Store mappings can be underspecified; any store of the task that doesn't have a mapping policy
   * will fall back to the default one.
   *
   * @param task Task to map
   * @param options Types of memories to which the stores can be mapped
   *
   * @return A vector of store mappings
   */
  virtual std::vector<StoreMapping> store_mappings(const Task& task,
                                                   const std::vector<StoreTarget>& options) = 0;
  /**
   * @brief Returns a tunable value
   *
   * @param tunable_id a tunable value id
   *
   * @return A tunable value in a `Scalar` object
   */
  virtual Scalar tunable_value(TunableID tunable_id) = 0;
};

}  // namespace legate::mapping
