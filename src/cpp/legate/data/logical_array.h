/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/logical_store.h>
#include <legate/data/shape.h>
#include <legate/mapping/mapping.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>
#include <legate/utilities/typedefs.h>

#include <optional>

/**
 * @file
 * @brief Class definition for legate::LogicalArray
 */

namespace legate::detail {

class LogicalArray;

}  // namespace legate::detail

namespace legate {

// This forward declaration is technically not sufficient. get_physical_array() returns a
// PhysicalArray and so we need the full definition. However, PhysicalArray holds a
// std::optional<LogicalArray> member, so we have a declaration cycle.
//
// But, it seems like any code that uses a LogicalArray ends up transitively including the
// physical_array.h header, and so this incorrect forward decl doesn't come to bite us (yet).
//
// Instead of having PhysicalArray hold a std::optional<LogicalArray>, we could have it hold a
// std::optional<SharedPtr<LogicalArrayImpl>> (assuming we pull LogicalArray::Impl out and
// rename it).
//
// But this is undesirable because it break encapsulation by leaking the implementation detail
// of LogicalArray to its physical counterpart. We also would have to do this for any child
// objects of LogicalArray, including when it undergoes transformation.
//
// So the least bad option is to do this fwd decl...
class PhysicalArray;

class ListLogicalArray;
class StringLogicalArray;

/**
 * @addtogroup data
 * @{
 */

/**
 * @brief A multi-dimensional array
 */
class LogicalArray {
 public:
  /**
   * @brief Returns the number of dimensions of the array.
   *
   * @return The number of dimensions
   */
  [[nodiscard]] std::uint32_t dim() const;

  /**
   * @brief Returns the element type of the array.
   *
   * @return `Type` of elements in the store
   */
  [[nodiscard]] Type type() const;

  /**
   * @brief Returns the `Shape` of the array.
   *
   * @return The store's `Shape`
   */
  [[nodiscard]] Shape shape() const;

  /**
   * @brief Returns the extents of the array.
   *
   * The call can block if the array is unbound
   *
   * @return The store's extents
   */
  [[nodiscard]] const tuple<std::uint64_t>& extents() const;

  /**
   * @brief Returns the number of elements in the array.
   *
   * The call can block if the array is unbound
   *
   * @return The number of elements in the store
   */
  [[nodiscard]] std::size_t volume() const;

  /**
   * @brief Indicates whether the array is unbound
   *
   * @return `true` if the array is unbound, `false` if it is normal
   */
  [[nodiscard]] bool unbound() const;

  /**
   * @brief Indicates whether the array is nullable
   *
   * @return `true` if the array is nullable, `false` otherwise
   */
  [[nodiscard]] bool nullable() const;

  /**
   * @brief Indicates whether the array has child arrays
   *
   * @return `true` if the array has child arrays, `false` otherwise
   */
  [[nodiscard]] bool nested() const;

  /**
   * @brief Returns the number of child sub-arrays
   *
   * @return Number of child sub-arrays
   */
  [[nodiscard]] std::uint32_t num_children() const;

  /**
   * @brief Adds an extra dimension to the array.
   *
   * The call can block if the array is unbound
   *
   * @param extra_dim Position for a new dimension
   * @param dim_size Extent of the new dimension
   *
   * @return A new array with an extra dimension
   *
   * @throw std::invalid_argument When `extra_dim` is not a valid dimension name
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  [[nodiscard]] LogicalArray promote(std::int32_t extra_dim, std::size_t dim_size) const;

  /**
   * @brief Projects out a dimension of the array.
   *
   * The call can block if the array is unbound
   *
   * @param dim Dimension to project out
   * @param index Index on the chosen dimension
   *
   * @return A new array with one fewer dimension
   *
   * @throw std::invalid_argument If `dim` is not a valid dimension name or `index` is out of bounds
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  [[nodiscard]] LogicalArray project(std::int32_t dim, std::int64_t index) const;

  /**
   * @brief Slices a contiguous sub-section of the array.
   *
   * The call can block if the array is unbound
   *
   * @param dim Dimension to slice
   * @param sl Slice descriptor
   *
   * @return A new array that corresponds to the sliced section
   *
   * @throw std::invalid_argument If `dim` is not a valid dimension name
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  [[nodiscard]] LogicalArray slice(std::int32_t dim, Slice sl) const;

  /**
   * @brief Reorders dimensions of the array.
   *
   * The call can block if the array is unbound
   *
   * @param axes Mapping from dimensions of the resulting array to those of the input
   *
   * @return A new array with the dimensions transposed
   *
   * @throw std::invalid_argument If any of the following happens: 1) The length of `axes` doesn't
   * match the array's dimension; 2) `axes` has duplicates; 3) Any axis in `axes` is an invalid
   * axis name.
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  [[nodiscard]] LogicalArray transpose(const std::vector<std::int32_t>& axes) const;

  /**
   * @brief Delinearizes a dimension into multiple dimensions.
   *
   * The call can block if the array is unbound
   *
   * @param dim Dimension to delinearize
   * @param sizes Extents for the resulting dimensions
   *
   * @return A new array with the chosen dimension delinearized
   *
   * @throw std::invalid_argument If `dim` is invalid for the array or `sizes` does not preserve
   * the extent of the chosen dimension
   * @throw std::runtime_error If the array or any of the sub-arrays is a list array
   */
  [[nodiscard]] LogicalArray delinearize(std::int32_t dim,
                                         const std::vector<std::uint64_t>& sizes) const;

  /**
   * @brief Returns the store of this array
   *
   * @return `LogicalStore`
   */
  [[nodiscard]] LogicalStore data() const;

  /**
   * @brief Returns the null mask of this array
   *
   * @return `LogicalStore`
   */
  [[nodiscard]] LogicalStore null_mask() const;

  /**
   * @brief Returns the sub-array of a given index
   *
   * @param index Sub-array index
   *
   * @return `LogicalArray`
   *
   * @throw std::invalid_argument If the array has no child arrays, or the array is an unbound
   * struct array
   * @throw std::out_of_range If the index is out of range
   */
  [[nodiscard]] LogicalArray child(std::uint32_t index) const;

  /**
   * @brief Creates a `PhysicalArray` for this `LogicalArray`
   *
   * This call blocks the client's control flow and fetches the data for the whole array to the
   * current node.
   *
   * When the target is `StoreTarget::FBMEM`, the data will be consolidated in the framebuffer of
   * the first GPU available in the scope.
   *
   * If no `target` is given, the runtime uses `StoreTarget::SOCKETMEM` if it exists and
   * `StoreTarget::SYSMEM` otherwise.
   *
   * If there already exists a physical array for a different memory target, that physical array
   * will be unmapped from memory and become invalid to access.
   *
   * @param target The type of memory in which the physical array would be created.
   *
   * @return A `PhysicalArray` of the `LogicalArray`
   *
   * @throw std::invalid_argument If no memory of the chosen type is available
   */
  [[nodiscard]] PhysicalArray get_physical_array(
    std::optional<mapping::StoreTarget> target = std::nullopt) const;

  /**
   * @brief Casts this array as a `ListLogicalArray`
   *
   * @return The array as a `ListLogicalArray`
   *
   * @throw std::invalid_argument If the array is not a list array
   */
  [[nodiscard]] ListLogicalArray as_list_array() const;

  /**
   * @brief Casts this array as a `StringLogicalArray`
   *
   * @return The array as a `StringLogicalArray`
   *
   * @throw std::invalid_argument If the array is not a string array
   */
  [[nodiscard]] StringLogicalArray as_string_array() const;

  /**
   * @brief Offload array to specified target memory.
   *
   * @param target_mem The target memory.
   *
   * Copies the array to the specified memory, if necessary, and marks it as the
   * most up-to-date copy, allowing the runtime to discard any copies in other
   * memories.
   *
   * Main usage is to free up space in one kind of memory by offloading resident
   * arrays and stores to another kind of memory. For example, after a GPU task
   * that reads or writes to an array, users can manually free up Legate's GPU
   * memory by offloading the array to host memory.
   *
   * All the stores that comprise the array are offloaded, i.e., the data store,
   * the null mask, and child arrays, etc.
   *
   * Currently, the runtime does not validate if the target memory has enough
   * capacity or free space at the point of launching or executing the offload
   * operation. The program will most likely crash if there isn't enough space in
   * the target memory. The user is therefore encouraged to offload to a memory
   * type that is likely to have sufficient space.
   *
   * This should not be treated as a prefetch call as it offers little benefit to
   * that end. The runtime will ensure that data for a task is resident in the
   * required memory before the task begins executing.
   *
   * If this array is backed by another array, e.g., if this array is a slice
   * or some other transform of another array, then both the arrays will be
   * offloaded due to being backed by the same memory.
   *
   * @snippet unit/logical_store/offload_to.cc offload-to-host
   *
   * @throws std::invalid_argument If Legate was not configured to
   * support `target_mem`.
   */
  void offload_to(mapping::StoreTarget target_mem) const;

  LogicalArray() = LEGATE_DEFAULT_WHEN_CYTHON;

  explicit LogicalArray(InternalSharedPtr<detail::LogicalArray> impl);

  virtual ~LogicalArray() noexcept;
  LogicalArray(const LogicalArray&)            = default;
  LogicalArray& operator=(const LogicalArray&) = default;
  LogicalArray(LogicalArray&&)                 = default;
  LogicalArray& operator=(LogicalArray&&)      = default;

  // NOLINTNEXTLINE(google-explicit-constructor) we want this?
  LogicalArray(const LogicalStore& store);
  LogicalArray(const LogicalStore& store, const LogicalStore& null_mask);

  [[nodiscard]] const SharedPtr<detail::LogicalArray>& impl() const;

 protected:
  class Impl;

  SharedPtr<Impl> impl_{nullptr};
};

/**
 * @brief Represents a logical array of variable-length lists.
 *
 * Each element of the array is itself a list, potentially of different length. For example, a
 * ListLogicalArray may represent:
 *
 * ```
 * [[a, b], [c], [d, e, f]]
 * ```
 *
 * This is stored using two arrays:
 *
 * 1. A descriptor array that defines the start and end indices of each sublist within the
 *    value data array. The descriptor array is stored as a series of `Rect<1>`s, where
 *    `lo` and `hi` members indicate the start and end of each range.
 * 2. A value data array (`vardata`) containing all list elements in a flattened form.
 *
 * For example:
 *
 * ```
 * descriptor: [ (0, 1), (2, 2), (3, 5) ]
 * vardata:    [ a, b, c, d, e, f ]
 * ```
 *
 * Where the mapping of `descriptor` to `vardata` follows:
 *
 * ```
 * descriptor     vardata
 * ----------     --------------------
 * (0, 1)    ---> [ a, b ]
 * (2, 2)    --->        [ c ]
 * (3, 5)    --->            [ d, e, f ]
 * ```
 *
 * @note The user can achieve the same effects of a `ListLogicalArray` themselves by applying
 * an image constraint (`image(Variable, Variable, ImageComputationHint)`) to two
 * `LogicalArray`s when passing them to a task. In that case `descriptor` would be
 * `var_function` while `vardata` would be `var_range`.
 */
class ListLogicalArray : public LogicalArray {
 public:
  /**
   * @brief Returns the sub-array for descriptors. Each element is a `Rect<1>` of start and end
   * indices for each subregion in `vardata`.
   *
   * @return Sub-array's for descriptors.
   */
  [[nodiscard]] LogicalArray descriptor() const;

  /**
   * @brief Returns the sub-array for variable size data.
   *
   * @return `LogicalArray` of variable sized data.
   */
  [[nodiscard]] LogicalArray vardata() const;

 private:
  friend class LogicalArray;

  explicit ListLogicalArray(InternalSharedPtr<detail::LogicalArray> impl);
};

/**
 * @brief A multi-dimensional array representing a collection of strings.
 *
 * This class is essentially a `ListLogicalArray` specialized for lists-of-strings data. The
 * member functions `offsets()` and `chars()` are directly analogous to
 * `ListLogicalArray::descriptor()` and `ListLogicalArray::vardata()`.
 *
 * The strings are stored in a compact form, accessible through `chars()`, while `offsets()`
 * gives the start and end indices for each sub-string.
 *
 * @see ListLogicalArray
 */
class StringLogicalArray : public LogicalArray {
 public:
  /**
   * @brief Returns the sub-array for offsets giving the bounds of each string.
   *
   * @return `LogicalArray` of offsets into this array.
   */
  [[nodiscard]] LogicalArray offsets() const;

  /**
   * @brief Returns the sub-array for characters of the strings.
   *
   * @return `LogicalArray` representing the characters of the strings.
   */
  [[nodiscard]] LogicalArray chars() const;

 private:
  friend class LogicalArray;

  explicit StringLogicalArray(InternalSharedPtr<detail::LogicalArray> impl);
};

/** @} */

}  // namespace legate
