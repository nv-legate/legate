/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/scalar.h>
#include <legate/mapping/store.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>

#include <iosfwd>
#include <memory>
#include <vector>

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

/**
 * @addtogroup mapping
 * @{
 */

class Task;

// NOTE: codes are chosen to reflect the precedence between the processor kinds in choosing target
// processors for tasks.

/**
 * @brief An enum class for task targets
 *
 * The enumerators of `TaskTarget` are ordered by their precedence; i.e., `GPU`, if available, is
 * chosen over `OMP` or `CPU, `OMP`, if available, is chosen over `CPU`.
 */
enum class TaskTarget : std::uint8_t {
  /**
   * @brief Indicates the task be mapped to a GPU
   */
  GPU,
  /**
   * @brief Indicates the task be mapped to an OpenMP processor
   */
  OMP,
  /**
   * @brief Indicates the task be mapped to a CPU
   */
  CPU,
};
static_assert(TaskTarget::GPU < TaskTarget::OMP);
static_assert(TaskTarget::OMP < TaskTarget::CPU);

std::ostream& operator<<(std::ostream& stream, const TaskTarget& target);

/**
 * @brief Enumerates the possible memory types a store may be mapped to.
 */
enum class StoreTarget : std::uint8_t {
  /**
   * @brief Indicates the store be mapped to the system memory (host memory)
   */
  SYSMEM,
  /**
   * @brief Indicates the store be mapped to the GPU framebuffer
   */
  FBMEM,
  /**
   * @brief Indicates the store be mapped to the pinned memory for zero-copy GPU accesses
   */
  ZCMEM,
  /**
   * @brief Indicates the store be mapped to the host memory closest to the target CPU
   */
  SOCKETMEM,
};

std::ostream& operator<<(std::ostream& stream, const StoreTarget& target);

/**
 * @brief An enum class for instance allocation policies
 */
enum class AllocPolicy : std::uint8_t {
  /**
   * @brief Indicates the store can reuse an existing instance
   */
  MAY_ALLOC,
  /**
   * @brief Indicates the store must be mapped to a fresh instance
   */
  MUST_ALLOC,
};

/**
 * @brief An enum class for instant layouts
 */
enum class InstLayout : std::uint8_t {
  /**
   * @brief Indicates the store must be mapped to an SOA instance
   */
  SOA,
  /**
   * @brief Indicates the store must be mapped to an AOS instance. No different than `SOA` in a
   * store mapping for a single store
   */
  AOS,
};

/**
 * @brief A descriptor for dimension ordering
 */
class DimOrdering {
 public:
  /**
   * @brief An enum class for kinds of dimension ordering
   */
  enum class Kind : std::uint8_t {
    /**
     * @brief Indicates the instance have C layout (i.e., the last dimension is the leading
     * dimension in the instance)
     */
    C,
    /**
     * @brief Indicates the instance have Fortran layout (i.e., the first dimension is the leading
     * dimension instance)
     */
    FORTRAN,
    /**
     * @brief Indicates the order of dimensions of the instance is manually specified
     */
    CUSTOM,
  };

  /**
   * @brief Creates a C ordering object
   *
   * @return A `DimOrdering` object
   */
  [[nodiscard]] static DimOrdering c_order();
  /**
   * @brief Creates a Fortran ordering object
   *
   * @return A `DimOrdering` object
   */
  [[nodiscard]] static DimOrdering fortran_order();
  /**
   * @brief Creates a custom ordering object
   *
   * @param dims A vector that stores the order of dimensions.
   *
   * @return A `DimOrdering` object
   */
  [[nodiscard]] static DimOrdering custom_order(std::vector<std::int32_t> dims);

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
  bool operator!=(const DimOrdering&) const;

  [[nodiscard]] const SharedPtr<detail::DimOrdering>& impl() const noexcept;

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
   * @brief If true, the runtime treats the instance as a redundant copy and marks it as collectible
   * as soon as the consumer task is done using it. In case where the program makes access to a
   * store through several different partitions, setting this flag will help reduce the memory
   * footprint by allowing the runtime to collect redundant instances eagerly.
   *
   * This flag has no effect when the instance is not freshly created for the task or is used for
   * updates.
   */
  bool redundant{false};

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
   * @brief Changes the value of `redundant`
   *
   * @param redundant A new value for the `redundant` field
   *
   * @return This instance mapping policy
   */
  [[nodiscard]] InstanceMappingPolicy& with_redundant(bool redundant) &;
  [[nodiscard]] InstanceMappingPolicy with_redundant(bool redundant) const&;
  [[nodiscard]] InstanceMappingPolicy&& with_redundant(bool redundant) &&;

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
   * @brief Changes the value of `redundant`
   *
   * @param redundant A new value for the `redundant` field
   */
  void set_redundant(bool redundant);

  bool operator==(const InstanceMappingPolicy&) const;
  bool operator!=(const InstanceMappingPolicy&) const;
};

/**
 * @brief A mapping policy for stores
 */
class StoreMapping {
 public:
  StoreMapping() noexcept;
  StoreMapping(StoreMapping&&) noexcept;
  StoreMapping& operator=(StoreMapping&&) noexcept;
  ~StoreMapping();

  explicit StoreMapping(std::unique_ptr<detail::StoreMapping> impl);

  /**
   * @brief Creates a mapping policy for the given store following the default mapping policy
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

  class ReleaseKey {
    ReleaseKey() = default;

    friend class detail::BaseMapper;
    friend StoreMapping;
  };

  // NOLINTNEXTLINE(readability-identifier-naming)
  [[nodiscard]] detail::StoreMapping* release_(ReleaseKey) noexcept;

 private:
  std::unique_ptr<detail::StoreMapping> impl_{};
};

/**
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
 * @brief An abstract class that defines Legate mapping APIs
 *
 * The APIs give Legate libraries high-level control on task and store mappings
 */
class Mapper {
 public:
  virtual ~Mapper() = default;
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
   * @brief Returns an upper bound for the amount of memory (in bytes), of a particular memory type,
   * allocated by a task via Legate allocators.
   *
   * All buffers created by ``create_buffer`` or ``create_output_buffer`` calls are drawn from this
   * allocation pool, and their aggregate size cannot exceed the upper bound returned from this call
   * (the program will crash otherwise). Any out-of-band memory allocations (e.g., those created by
   * ``malloc`` or ``cudaMalloc``) invisible to Legate are not subject to this pool bound.
   *
   * This callback is invoked only for task variants that are registered with ``has_allocations``
   * being ``true``.
   *
   * @param task Task to map
   * @param memory_kind Type of memory in which the memory pool is created
   *
   * @return A memory pool size; returning ``std::nullopt`` means the total size is unknown.
   */
  [[nodiscard]] virtual std::optional<std::size_t> allocation_pool_size(
    const Task& task, StoreTarget memory_kind) = 0;
  /**
   * @brief Returns a tunable value
   *
   * @param tunable_id a tunable value id
   *
   * @return A tunable value in a `Scalar` object
   */
  [[nodiscard]] virtual Scalar tunable_value(TunableID tunable_id) = 0;
};

/** @} */

}  // namespace legate::mapping

#include <legate/mapping/mapping.inl>
