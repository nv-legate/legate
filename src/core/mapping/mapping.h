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

#include "core/data/scalar.h"
#include "core/mapping/store.h"
#include "core/utilities/internal_shared_ptr.h"
#include "core/utilities/shared_ptr.h"

#include <iosfwd>
#include <memory>
#include <vector>

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
enum class TaskTarget : std::int32_t {
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
enum class StoreTarget : std::int32_t {
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
enum class AllocPolicy : std::int32_t {
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
enum class InstLayout : std::int32_t {
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
class DimOrdering {
 public:
  /**
   * @brief An enum class for kinds of dimension ordering
   */
  enum class Kind : std::int32_t {
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
  static DimOrdering custom_order(std::vector<std::int32_t> dims);

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
  void set_custom_order(std::vector<std::int32_t> dims);

  /**
   * @brief Dimension ordering type
   */
  [[nodiscard]] Kind kind() const;
  /**
   * @brief Dimension list. Used only when the `kind` is `CUSTOM`.
   */
  [[nodiscard]] std::vector<std::int32_t> dimensions() const;

  bool operator==(const DimOrdering&) const;

  [[nodiscard]] const detail::DimOrdering* impl() const noexcept;

  DimOrdering()                                  = default;
  DimOrdering(const DimOrdering&)                = default;
  DimOrdering& operator=(const DimOrdering&)     = default;
  DimOrdering(DimOrdering&&) noexcept            = default;
  DimOrdering& operator=(DimOrdering&&) noexcept = default;
  ~DimOrdering() noexcept;

 private:
  explicit DimOrdering(InternalSharedPtr<detail::DimOrdering> impl);

  SharedPtr<detail::DimOrdering> impl_{c_order().impl_};
};

/**
 * @ingroup mapping
 * @brief A descriptor for instance mapping policy
 */
class InstanceMappingPolicy {
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

  /**
   * @brief Changes the store target
   *
   * @param target A new store target
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] InstanceMappingPolicy& with_target(StoreTarget target) &;
  [[nodiscard]] InstanceMappingPolicy with_target(StoreTarget target) const&;
  [[nodiscard]] InstanceMappingPolicy&& with_target(StoreTarget target) &&;

  /**
   * @brief Changes the allocation policy
   *
   * @param allocation A new allocation policy
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] InstanceMappingPolicy& with_allocation_policy(AllocPolicy allocation) &;
  [[nodiscard]] InstanceMappingPolicy with_allocation_policy(AllocPolicy allocation) const&;
  [[nodiscard]] InstanceMappingPolicy&& with_allocation_policy(AllocPolicy allocation) &&;

  /**
   * @brief Changes the instance layout
   *
   * @param layout A new instance layout
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] InstanceMappingPolicy& with_instance_layout(InstLayout layout) &;
  [[nodiscard]] InstanceMappingPolicy with_instance_layout(InstLayout layout) const&;
  [[nodiscard]] InstanceMappingPolicy&& with_instance_layout(InstLayout layout) &&;

  /**
   * @brief Changes the dimension ordering
   *
   * @param ordering A new dimension ordering
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] InstanceMappingPolicy& with_ordering(DimOrdering ordering) &;
  [[nodiscard]] InstanceMappingPolicy with_ordering(DimOrdering ordering) const&;
  [[nodiscard]] InstanceMappingPolicy&& with_ordering(DimOrdering ordering) &&;

  /**
   * @brief Changes the value of `exact`
   *
   * @param exact A new value for the `exact` field
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] InstanceMappingPolicy& with_exact(bool exact) &;
  [[nodiscard]] InstanceMappingPolicy with_exact(bool exact) const&;
  [[nodiscard]] InstanceMappingPolicy&& with_exact(bool exact) &&;

  /**
   * @brief Changes the store target
   *
   * @param target A new store target
   */
  void set_target(StoreTarget target);
  /**
   * @brief Changes the allocation policy
   *
   * @param allocation A new allocation policy
   */
  void set_allocation_policy(AllocPolicy allocation);
  /**
   * @brief Changes the instance layout
   *
   * @param layout A new instance layout
   */
  void set_instance_layout(InstLayout layout);
  /**
   * @brief Changes the dimension ordering
   *
   * @param ordering A new dimension ordering
   */
  void set_ordering(DimOrdering ordering);
  /**
   * @brief Changes the value of `exact`
   *
   * @param exact A new value for the `exact` field
   */
  void set_exact(bool exact);

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
  [[nodiscard]] bool subsumes(const InstanceMappingPolicy& other) const;

  bool operator==(const InstanceMappingPolicy&) const;
  bool operator!=(const InstanceMappingPolicy&) const;
};

/**
 * @ingroup mapping
 * @brief A mapping policy for stores
 */
class StoreMapping {
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
  [[nodiscard]] static StoreMapping default_mapping(const Store& store,
                                                    StoreTarget target,
                                                    bool exact = false);
  /**
   * @brief Creates a mapping policy for the given store using the instance mapping policy
   *
   * @param store Target store for the mapping policy
   * @param policy Instance mapping policy to apply
   *
   * @return A store mapping
   */
  [[nodiscard]] static StoreMapping create(const Store& store, InstanceMappingPolicy&& policy);

  /**
   * @brief Creates a mapping policy for the given set of stores using the instance mapping policy
   *
   * @param stores Target stores for the mapping policy
   * @param policy Instance mapping policy to apply
   *
   * @return A store mapping
   */
  [[nodiscard]] static StoreMapping create(const std::vector<Store>& stores,
                                           InstanceMappingPolicy&& policy);

  /**
   * @brief Returns the instance mapping policy of this `StoreMapping` object
   *
   * @return A reference to the `InstanceMappingPolicy` object
   */
  [[nodiscard]] InstanceMappingPolicy& policy();
  /**
   * @brief Returns the instance mapping policy of this `StoreMapping` object
   *
   * @return A reference to the `InstanceMappingPolicy` object
   */
  [[nodiscard]] const InstanceMappingPolicy& policy() const;

  /**
   * @brief Returns the store for which this `StoreMapping` object describes a mapping policy.
   *
   * If the policy is for multiple stores, the first store added to this policy will be returned;
   *
   * @return A `Store` object
   */
  [[nodiscard]] Store store() const;
  /**
   * @brief Returns all the stores for which this `StoreMapping` object describes a mapping policy
   *
   * @return A vector of `Store` objects
   */
  [[nodiscard]] std::vector<Store> stores() const;

  /**
   * @brief Adds a store to this `StoreMapping` object
   *
   * @param store Store to add
   */
  void add_store(const Store& store);

  [[nodiscard]] const detail::StoreMapping* impl() const noexcept;

  StoreMapping() = default;

 private:
  friend class detail::BaseMapper;
  detail::StoreMapping* release() noexcept;

  explicit StoreMapping(detail::StoreMapping* impl) noexcept;

  // Work-around for using unique_ptr for PIMPL. unique_ptr requires the type to be defined in
  // its destructor, as required by delete. This is a problem because the implicitly declared
  // destructor/move assignment/move constructor calls the unique_ptr destructor, and since we
  // don't define them, they are implicitly defined inline above.
  //
  // One solution then is to manually define these functions out-of-line (can still be done
  // trivially, i.e. StoreMapping::~StoreMapping() = default), but the whole point of using
  // unique_ptr is that we *don't* want to write these functions!
  //
  // The better solution then is to hide the call to delete behind a custom deleter. Hence
  // StoreMappingImplDeleter.
  class StoreMappingImplDeleter {
   public:
    void operator()(detail::StoreMapping* ptr) const noexcept;
  };

  std::unique_ptr<detail::StoreMapping, StoreMappingImplDeleter> impl_{};
};

/**
 * @ingroup mapping
 * @brief An abstract class that defines machine query APIs
 */
class MachineQueryInterface {
 public:
  virtual ~MachineQueryInterface() = default;
  /**
   * @brief Returns local CPUs
   *
   * @return A vector of processors
   */
  [[nodiscard]] virtual const std::vector<Processor>& cpus() const = 0;
  /**
   * @brief Returns local GPUs
   *
   * @return A vector of processors
   */
  [[nodiscard]] virtual const std::vector<Processor>& gpus() const = 0;
  /**
   * @brief Returns local OpenMP processors
   *
   * @return A vector of processors
   */
  [[nodiscard]] virtual const std::vector<Processor>& omps() const = 0;
  /**
   * @brief Returns the total number of nodes
   *
   * @return Total number of nodes
   */
  [[nodiscard]] virtual std::uint32_t total_nodes() const = 0;
};

/**
 * @ingroup mapping
 * @brief An abstract class that defines Legate mapping APIs
 *
 * The APIs give Legate libraries high-level control on task and store mappings
 */
class Mapper {
 public:
  virtual ~Mapper() = default;
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
  [[nodiscard]] virtual TaskTarget task_target(const Task& task,
                                               const std::vector<TaskTarget>& options) = 0;
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
  [[nodiscard]] virtual std::vector<StoreMapping> store_mappings(
    const Task& task, const std::vector<StoreTarget>& options) = 0;
  /**
   * @brief Returns a tunable value
   *
   * @param tunable_id a tunable value id
   *
   * @return A tunable value in a `Scalar` object
   */
  [[nodiscard]] virtual Scalar tunable_value(TunableID tunable_id) = 0;
};

}  // namespace legate::mapping

#include "core/mapping/mapping.inl"
