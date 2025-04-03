/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/shape.h>
#include <legate/data/slice.h>
#include <legate/mapping/mapping.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>

#include <optional>
#include <utility>
#include <vector>

/**
 * @file
 * @brief Class definition for legate::LogicalStore and
 * legate::LogicalStorePartition
 */

namespace legate::detail {
class LogicalArray;
class LogicalStore;
class LogicalStorePartition;
class Storage;
}  // namespace legate::detail

namespace legate {

/**
 * @addtogroup data
 * @{
 */

// This forward declaration is technically not sufficient. get_physical_array() returns a
// PhysicalStore and so we need the full definition. However, PhysicalStore holds a
// std::optional<LogicalStore> member, so we have a declaration cycle.
//
// But, it seems like any code that uses a LogicalStore ends up transitively including the
// physical_store.h header, and so this incorrect forward decl doesn't come to bite us (yet).
//
// Instead of having PhysicalStore hold a std::optional<LogicalStore>, we could have it hold a
// std::optional<SharedPtr<LogicalStoreImpl>> (assuming we pull LogicalStore::Impl out and
// rename it).
//
// But this is undesirable because it break encapsulation by leaking the implementation detail
// of LogicalStore to its physical counterpart. We also would have to do this for any child
// objects of LogicalStore, including when it undergoes transformation.
//
// So the least bad option is to do this fwd decl...
class PhysicalStore;

class LogicalStorePartition;
class Runtime;

/**
 * @brief A multi-dimensional data container
 *
 * `LogicalStore` is a multi-dimensional data container for fixed-size elements. Stores are
 * internally partitioned and distributed across the system. By default, Legate clients need
 * not create nor maintain the partitions explicitly, and the Legate runtime is responsible
 * for managing them. Legate clients can control how stores should be partitioned for a given
 * task by attaching partitioning constraints to the task (see the constraint module for
 * partitioning constraint APIs).
 *
 * Each `LogicalStore` object is a logical handle to the data and is not immediately associated
 * with a physical allocation. To access the data, a client must "map" the store to a physical
 * store (`PhysicalStore`). A client can map a store by passing it to a task, in which case the
 * task body can see the allocation, or calling `LogicalStore::get_physical_store()`, which
 * gives the client a handle to the physical allocation (see `PhysicalStore` for details about
 * physical stores).
 *
 * Normally, a `LogicalStore` gets a fixed `Shape` upon creation. However, there is a special
 * type of logical stores called "unbound" stores whose shapes are unknown at creation
 * time. (see `Runtime` for the logical store creation API.) The shape of an unbound store is
 * determined by a task that first updates the store; upon the submission of the task, the
 * `LogicalStore` becomes a normal store. Passing an unbound store as a read-only argument or
 * requesting a `PhysicalStore` of an unbound store are invalid.
 *
 * One consequence due to the nature of unbound stores is that querying the shape of a previously
 * unbound store can block the client's control flow for an obvious reason; to know the shape of
 * the `LogicalStore` whose `Shape` was unknown at creation time, the client must wait until the
 * updater task to finish. However, passing a previously unbound store to a downstream operation can
 * be non-blocking, as long as the operation requires no changes in the partitioning and mapping for
 * the `LogicalStore`.
 */
class LogicalStore {
  friend class Runtime;
  friend class LogicalArray;
  friend class LogicalStorePartition;

 public:
  explicit LogicalStore(InternalSharedPtr<detail::LogicalStore> impl);

  /**
   * @brief Returns the number of dimensions of the store.
   *
   * @return The number of dimensions
   */
  [[nodiscard]] std::uint32_t dim() const;

  /**
   * @brief Indicates whether the store's storage is optimized for scalars
   *
   * @return true The store is backed by a scalar storage
   * @return false The store is a backed by a normal region storage
   */
  [[nodiscard]] bool has_scalar_storage() const;

  /**
   * @brief Indicates whether this store overlaps with a given store
   *
   * @return true The stores overlap
   * @return false The stores are disjoint
   */
  [[nodiscard]] bool overlaps(const LogicalStore& other) const;

  /**
   * @brief Returns the element type of the store.
   *
   * @return `Type` of elements in the store
   */
  [[nodiscard]] Type type() const;

  /**
   * @brief Returns the shape of the array.
   *
   * @return The store's `Shape`
   */
  [[nodiscard]] Shape shape() const;

  /**
   * @brief Returns the extents of the store.
   *
   * The call can block if the store is unbound
   *
   * @return The store's extents
   */
  [[nodiscard]] const tuple<std::uint64_t>& extents() const;

  /**
   * @brief Returns the number of elements in the store.
   *
   * The call can block if the store is unbound
   *
   * @return The number of elements in the store
   */
  [[nodiscard]] std::size_t volume() const;

  /**
   * @brief Indicates whether the store is unbound
   *
   * @return `true` if the store is unbound, `false` otherwise
   */
  [[nodiscard]] bool unbound() const;

  /**
   * @brief Indicates whether the store is transformed
   *
   * @return `true` if the store is transformed, `false` otherwise
   */
  [[nodiscard]] bool transformed() const;

  /**
   * @brief Reinterpret the underlying data of a `LogicalStore` byte-for-byte as another type.
   *
   * @param type The new type to interpret the data as.
   *
   * @return The reinterpreted store.
   *
   * The size and alignment of the new type must match that of the existing type.
   *
   * The reinterpreted store will share the same underlying storage as the original, and
   * therefore any writes to one will also be reflected in the other. No type conversions of
   * any kind are performed across the stores, the bytes are interpreted as-is. In effect,
   * if one were to model a `LogicalStore` as a pointer to an array, then this routine is
   * equivalent to `reinterpret_cast`-ing the pointer.
   *
   * Example:
   * @snippet unit/logical_store/reinterpret_as.cc Reinterpret store data
   *
   * @throw std::invalid_argument If the size (in bytes) of the new type does not match that of
   * the old type.
   * @throw std::invalid_argument If the alignment of the new type does not match that of the
   * old type.
   */
  [[nodiscard]] LogicalStore reinterpret_as(const Type& type) const;

  /**
   * @brief Adds an extra dimension to the store.
   *
   * Value of `extra_dim` decides where a new dimension should be added, and each dimension
   * @f$i@f$, where @f$i@f$ >= `extra_dim`, is mapped to dimension @f$i+1@f$ in a returned store.
   * A returned store provides a view to the input store where the values are broadcasted along
   * the new dimension.
   *
   * For example, for a 1D store `A` contains `[1, 2, 3]`, `A.promote(0, 2)` yields a store
   * equivalent to:
   *
   * @code{.unparsed}
   * [[1, 2, 3],
   *  [1, 2, 3]]
   * @endcode
   *
   * whereas `A.promote(1, 2)` yields:
   *
   * @code{.unparsed}
   * [[1, 1],
   *  [2, 2],
   *  [3, 3]]
   * @endcode
   *
   * The call can block if the store is unbound
   *
   * @param extra_dim Position for a new dimension
   * @param dim_size Extent of the new dimension
   *
   * @return A new store with an extra dimension
   *
   * @throw std::invalid_argument When `extra_dim` is not a valid dimension name
   */
  [[nodiscard]] LogicalStore promote(std::int32_t extra_dim, std::size_t dim_size) const;

  /**
   * @brief Projects out a dimension of the store.
   *
   * Each dimension @f$i@f$, where @f$i@f$ > `dim`, is mapped to dimension @f$i-1@f$ in a returned
   * store. A returned store provides a view to the input store where the values are on hyperplane
   * @f$x_\mathtt{dim} = \mathtt{index}@f$.
   *
   * For example, if a 2D store `A` contains `[[1, 2], [3, 4]]`, `A.project(0, 1)` yields a store
   * equivalent to `[3, 4]`, whereas `A.project(1, 0)` yields `[1, 3]`.
   *
   * The call can block if the store is unbound
   *
   * @param dim Dimension to project out
   * @param index Index on the chosen dimension
   *
   * @return A new store with one fewer dimension
   *
   * @throw std::invalid_argument If `dim` is not a valid dimension name or `index` is out of bounds
   */
  [[nodiscard]] LogicalStore project(std::int32_t dim, std::int64_t index) const;

  /**
   * @brief Slices a contiguous sub-section of the store.
   *
   * For example, consider a 2D store `A`:
   *
   * @code{.unparsed}
   * [[1, 2, 3],
   *  [4, 5, 6],
   *  [7, 8, 9]]
   * @endcode
   *
   * A slicing `A.slice(0, legate::Slice{1})` yields
   *
   * @code{.unparsed}
   * [[4, 5, 6],
   *  [7, 8, 9]]
   * @endcode
   *
   * The result store will look like this on a different slicing call
   * `A.slice(1, legate::Slice{legate::Slice::OPEN, 2})`:
   *
   * @code{.unparsed}
   * [[1, 2],
   *  [4, 5],
   *  [7, 8]]
   * @endcode
   *
   * Finally, chained slicing calls
   *
   * @code{.cpp}
   * A.slice(0, legate::Slice{1})
   *  .slice(1, legate::Slice{legate::Slice::OPEN, 2})
   * @endcode
   *
   * results in:
   *
   * @code{.unparsed}
   * [[4, 5],
   *  [7, 8]]
   * @endcode
   *
   * The call can block if the store is unbound
   *
   * @param dim Dimension to slice
   * @param sl `Slice` descriptor
   *
   * @return A new store that corresponds to the sliced section
   *
   * @throw std::invalid_argument If `dim` is not a valid dimension name
   */
  [[nodiscard]] LogicalStore slice(std::int32_t dim, Slice sl) const;

  /**
   * @brief Reorders dimensions of the store.
   *
   * Dimension @f$i@f$i of the resulting store is mapped to dimension `axes[i]` of the input store.
   *
   * For example, for a 3D store `A`
   *
   * @code{.unparsed}
   * [[[1, 2],
   *   [3, 4]],
   *  [[5, 6],
   *   [7, 8]]]
   * @endcode
   *
   * transpose calls `A.transpose({1, 2, 0})` and `A.transpose({2, 1, 0})` yield the following
   * stores, respectively:
   *
   * @code{.unparsed}
   * [[[1, 5],
   *   [2, 6]],
   *  [[3, 7],
   *   [4, 8]]]
   * @endcode
   *
   * @code{.unparsed}
   * [[[1, 5],
   *  [3, 7]],
   *
   *  [[2, 6],
   *   [4, 8]]]
   * @endcode
   *
   * The call can block if the store is unbound
   *
   * @param axes Mapping from dimensions of the resulting store to those of the input
   *
   * @return A new store with the dimensions transposed
   *
   * @throw std::invalid_argument If any of the following happens: 1) The length of `axes` doesn't
   * match the store's dimension; 2) `axes` has duplicates; 3) Any axis in `axes` is an invalid
   * axis name.
   */
  [[nodiscard]] LogicalStore transpose(std::vector<std::int32_t>&& axes) const;

  /**
   * @brief Delinearizes a dimension into multiple dimensions.
   *
   * Each dimension @f$i@f$ of the store, where @f$i >@f$ `dim`, will be mapped to dimension
   * @f$i+N@f$ of the resulting store, where @f$N@f$ is the length of `sizes`. A delinearization
   * that does not preserve the size of the store is invalid.
   *
   * For example, consider a 2D store `A`
   *
   * @code{.unparsed}
   * [[1, 2, 3, 4],
   *  [5, 6, 7, 8]]
   * @endcode
   *
   * A delinearizing call `A.delinearize(1, {2, 2}))` yields:
   *
   * @code{.unparsed}
   * [[[1, 2],
   *   [3, 4]],
   *
   *  [[5, 6],
   *   [7, 8]]]
   * @endcode
   *
   * Unlike other transformations, delinearization is not an affine transformation. Due to this
   * nature, delinearized stores can raise `legate::NonInvertibleTransformation` in places where
   * they cannot be used.
   *
   * The call can block if the store is unbound
   *
   * @param dim Dimension to delinearize
   * @param sizes Extents for the resulting dimensions
   *
   * @return A new store with the chosen dimension delinearized
   *
   * @throw std::invalid_argument If `dim` is invalid for the store or `sizes` does not preserve
   * the extent of the chosen dimension
   */
  [[nodiscard]] LogicalStore delinearize(std::int32_t dim, std::vector<std::uint64_t> sizes) const;

  /**
   * @brief Creates a tiled partition of the store
   *
   * The call can block if the store is unbound
   *
   * @param tile_shape Shape of tiles
   *
   * @return A store partition
   */
  [[nodiscard]] LogicalStorePartition partition_by_tiling(
    std::vector<std::uint64_t> tile_shape) const;

  /**
   * @brief Creates a `PhysicalStore` for this `LogicalStore`
   *
   * This call blocks the client's control flow and fetches the data for the whole store to the
   * current node.
   *
   * When the target is `StoreTarget::FBMEM`, the data will be consolidated in the framebuffer of
   * the first GPU available in the scope.
   *
   * If no `target` is given, the runtime uses `StoreTarget::SOCKETMEM` if it exists and
   * `StoreTarget::SYSMEM` otherwise.
   *
   * If there already exists a physical store for a different memory target, that physical store
   * will be unmapped from memory and become invalid to access.
   *
   * @param target The type of memory in which the physical store would be created.
   *
   * @return A `PhysicalStore` of the `LogicalStore`
   *
   * @throw std::invalid_argument If no memory of the chosen type is available
   */
  [[nodiscard]] PhysicalStore get_physical_store(
    std::optional<mapping::StoreTarget> target = std::nullopt) const;

  /**
   * @brief Detach a store from its attached memory
   *
   * This call will wait for all operations that use the store (or any sub-store) to complete.
   *
   * After this call returns, it is safe to deallocate the attached external allocation. If the
   * allocation was mutable, the contents would be up-to-date upon the return. The contents of the
   * store are invalid after that point.
   */
  void detach();

  /**
   * @brief Offload store to specified target memory.
   *
   * @param target_mem The target memory.
   *
   * @see `LogicalArray::offload_to()`.
   */
  void offload_to(mapping::StoreTarget target_mem);

  /**
   * @brief Determine whether two stores refer to the same memory.
   *
   * @param other The `LogicalStore` to compare with.
   *
   * @return `true` if two stores cover the same underlying memory region, `false` otherwise.
   *
   * This routine can be used to determine whether two seemingly unrelated stores refer to the
   * same logical memory region, including through possible transformations in either `this` or
   * `other`.
   *
   * The user should note that some transformations *do* modify the underlying storage. For
   * example, the store produced by slicing will *not* share the same storage as its parent,
   * and this routine will return false for it:
   *
   * @snippet unit/logical_store/equal_storage.cc Store::equal_storage: Comparing sliced stores
   *
   * Transposed stores, on the other hand, still share the same storage, and hence this routine
   * will return true for them:
   *
   * @snippet unit/logical_store/equal_storage.cc Store::equal_storage: Comparing transposed stores
   */
  [[nodiscard]] bool equal_storage(const LogicalStore& other) const;

  [[nodiscard]] std::string to_string() const;

  [[nodiscard]] const SharedPtr<detail::LogicalStore>& impl() const;

  LogicalStore() = LEGATE_DEFAULT_WHEN_CYTHON;

  LogicalStore(const LogicalStore& other)                = default;
  LogicalStore& operator=(const LogicalStore& other)     = default;
  LogicalStore(LogicalStore&& other) noexcept            = default;
  LogicalStore& operator=(LogicalStore&& other) noexcept = default;
  ~LogicalStore() noexcept;

 private:
  class Impl;

  SharedPtr<Impl> impl_{};
};

class LogicalStorePartition {
 public:
  explicit LogicalStorePartition(InternalSharedPtr<detail::LogicalStorePartition> impl);

  [[nodiscard]] LogicalStore store() const;
  [[nodiscard]] const tuple<std::uint64_t>& color_shape() const;
  [[nodiscard]] LogicalStore get_child_store(const tuple<std::uint64_t>& color) const;

  [[nodiscard]] const SharedPtr<detail::LogicalStorePartition>& impl() const;

  LogicalStorePartition()                                                  = default;
  LogicalStorePartition(const LogicalStorePartition& other)                = default;
  LogicalStorePartition& operator=(const LogicalStorePartition& other)     = default;
  LogicalStorePartition(LogicalStorePartition&& other) noexcept            = default;
  LogicalStorePartition& operator=(LogicalStorePartition&& other) noexcept = default;
  ~LogicalStorePartition() noexcept;

 private:
  class Impl;
  InternalSharedPtr<Impl> impl_{};
};

/** @} */

}  // namespace legate
