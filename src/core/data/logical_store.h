/* Copyright 2021 NVIDIA Corporation
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

#include <memory>
#include <valarray>

#include "legion.h"

#include "core/data/shape.h"
#include "core/data/slice.h"
#include "core/data/transform.h"
#include "core/type/type_info.h"
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition for legate::LogicalStore and
 * legate::LogicalStorePartition
 */

namespace legate::mapping {
class MachineDesc;
}  // namespace legate::mapping

namespace legate::detail {
class LogicalStore;
class LogicalStorePartition;
}  // namespace legate::detail

namespace legate {

class LibraryContext;
class LogicalStorePartition;
class Partition;
class Runtime;
class Store;

/**
 * @ingroup data
 *
 * @brief A multi-dimensional data container
 *
 * `LogicalStore` is a multi-dimensional data container for fixed-size elements. Stores are
 * internally partitioned and distributed across the system. By default, Legate clients need
 * not create nor maintain the partitions explicitly, and the Legate runtime is responsible
 * for managing them. Legate clients can control how stores should be partitioned for a given
 * task by attaching partitioning constraints to the task (see the constraint module for
 * partitioning constraint APIs).
 *
 * Each logical store object is a logical handle to the data and is not immediately associated
 * with a physical allocation. To access the data, a client must `map` the store to a physical
 * store (`Store`). A client can map a store by passing it to a task, in which case the task
 * body can see the allocation, or calling `get_physical_store`, which gives the client a handle
 * to the physical allocation (see `Store` for details about physical stores).
 *
 * Normally, a logical store gets a fixed shape upon creation. However, there is a special type of
 * logical stores called `unbound` stores whose shapes are unknown at creation time. (see `Runtime`
 * for the logical store creation API.) The shape of an unbound store is determined by a task that
 * first updates the store; upon the submission of the task, the logical store becomes a normal
 * store. Passing an unbound store as a read-only argument or requesting a physical store of an
 * unbound store are invalid.
 *
 * One consequence due to the nature of unbound stores is that querying the shape of a previously
 * unbound store can block the client's control flow for an obvious reason; to know the shape of
 * the logical store whose shape was unknown at creation time, the client must wait until the
 * updater task to finish. However, passing a previously unbound store to a downstream operation can
 * be non-blocking, as long as the operation requires no changes in the partitioning and mapping for
 * the logical store.
 */
class LogicalStore {
 private:
  friend class Runtime;
  friend class LogicalStorePartition;
  LogicalStore(std::shared_ptr<detail::LogicalStore>&& impl);

 public:
  LogicalStore(const LogicalStore& other)            = default;
  LogicalStore& operator=(const LogicalStore& other) = default;

 public:
  LogicalStore(LogicalStore&& other)            = default;
  LogicalStore& operator=(LogicalStore&& other) = default;

 public:
  /**
   * @brief Returns the number of dimensions of the store.
   *
   * @return The number of dimensions
   */
  int32_t dim() const;
  /**
   * @brief Returns the element type of the store.
   *
   * @return Type of elements in the store
   */
  const Type& type() const;
  /**
   * @brief Returns the shape of the store.
   *
   * Flushes the scheduling window if the store is unbound and has no shape assigned.
   *
   * @return The store's shape
   */
  const Shape& extents() const;
  size_t volume() const;
  /**
   * @brief Indicates whether the store is unbound
   *
   * @return true The store is unbound
   * @return false The store is normal
   */
  bool unbound() const;
  /**
   * @brief Indicates whether the store is transformed
   *
   * @return true The store is transformed
   * @return false The store is not transformed
   */
  bool transformed() const;

 public:
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
   * @param extra_dim Position for a new dimension
   * @param dim_size Extent of the new dimension
   *
   * @return A new store with an extra dimension
   *
   * @throw std::invalid_argument When `extra_dim` is not a valid dimension name
   */
  LogicalStore promote(int32_t extra_dim, size_t dim_size) const;
  /**
   * @brief Projects out a dimension of the store.
   *
   * Each dimension @f$@f$, where @f$i@f$ > `dim`, is mapped to dimension @f$i-1@f$ in a returned
   * store. A returned store provides a view to the input store where the values are on hyperplane
   * @f$x_\mathtt{dim} = \mathtt{index}@f$.
   *
   * For example, if a 2D store `A` contains `[[1, 2], [3, 4]]`, `A.project(0, 1)` yields a store
   * equivalent to `[3, 4]`, whereas `A.project(1, 0)` yields `[1, 3]`.
   *
   * @param dim Dimension to project out
   * @param index Index on the chosen dimension
   *
   * @return A new store with one fewer dimension
   *
   * @throw std::invalid_argument If `dim` is not a valid dimension name or `index` is out of bounds
   */
  LogicalStore project(int32_t dim, int64_t index) const;
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
   * A slicing `A.slice(0, legate::Slice(1))` yields
   *
   * @code{.unparsed}
   * [[4, 5, 6],
   *  [7, 8, 9]]
   * @endcode
   *
   * The result store will look like this on a different slicing call
   * `A.slice(1, legate::Slice(legate::Slice::OPEN, 2))`:
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
   * A.slice(0, legate::Slice(1)).slice(1, legate::Slice(legate::Slice::OPEN, 2))
   * @endcode
   *
   * results in:
   *
   * @code{.unparsed}
   * [[4, 5],
   *  [7, 8]]
   * @endcode
   *
   * @param dim Dimension to slice
   * @param sl Slice descriptor
   *
   * @return A new store that correponds to the sliced section
   *
   * @throw std::invalid_argument If `dim` is not a valid dimension name
   */
  LogicalStore slice(int32_t dim, Slice sl) const;
  /**
   * @brief Reorders dimensions of the store.
   *
   * Dimension `i` of the resulting store is mapped to dimension `axes[i]` of the input store.
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
   * @code(.unparsed}
   * [[[1, 5],
   *  [3, 7]],
   *
   *  [[2, 6],
   *   [4, 8]]]
   * @endcode
   *
   * @param axes Mapping from dimensions of the resulting store to those of the input
   *
   * @return A new store with the dimensions transposed
   *
   * @throw std::invalid_argument If any of the following happens: 1) The length of `axes` doesn't
   * match the store's dimension; 2) `axes` has duplicates; 3) Any axis in `axes` is an invalid
   * axis name.
   */
  LogicalStore transpose(std::vector<int32_t>&& axes) const;
  /**
   * @brief Delinearizes a dimension into multiple dimensions.
   *
   * Each dimension @f$i@f$ of the store, where @f$i > @f$`dim`, will be mapped to dimension
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
   * @param dim Dimension to delinearize
   * @param sizes Extents for the resulting dimensions
   *
   * @return A new store with the chosen dimension delinearized
   *
   * @throw std::invalid_argument If `dim` is invalid for the store or `sizes` does not preserve
   * the extent of the chosen dimenison
   */
  LogicalStore delinearize(int32_t dim, std::vector<int64_t>&& sizes) const;

 public:
  /**
   * @brief Creates a tiled partition of the store
   *
   * @param tile_shape Shape of tiles
   *
   * @return A store partition
   */
  LogicalStorePartition partition_by_tiling(std::vector<size_t> tile_shape) const;

 public:
  /**
   * @brief Creates a physical store for this logical store
   *
   * This call blocks the client's control flow. And it fetches the data for the whole store on
   * a single node.
   *
   * @return A physical store of the logical store
   */
  std::shared_ptr<Store> get_physical_store();

 public:
  void set_key_partition(const mapping::MachineDesc& machine, const Partition* partition);

 public:
  std::shared_ptr<detail::LogicalStore> impl() const { return impl_; }

 private:
  std::shared_ptr<detail::LogicalStore> impl_{nullptr};
};

class LogicalStorePartition {
 private:
  friend class LogicalStore;
  LogicalStorePartition(std::shared_ptr<detail::LogicalStorePartition>&& impl);

 public:
  LogicalStore store() const;
  std::shared_ptr<Partition> partition() const;

 private:
  std::shared_ptr<detail::LogicalStorePartition> impl_{nullptr};
};

}  // namespace legate
