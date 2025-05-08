/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate_defines.h>

#include <legate/data/buffer.h>
#include <legate/data/inline_allocation.h>
#include <legate/data/logical_store.h>
#include <legate/mapping/mapping.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

/**
 * @file
 * @brief Class definition for legate::PhysicalStore
 */

namespace legate::detail {
class PhysicalStore;
}  // namespace legate::detail

namespace legate {

/**
 * @addtogroup data
 * @{
 */

class PhysicalArray;

#define LEGATE_TRUE_WHEN_DEBUG LEGATE_DEFINED(LEGATE_USE_DEBUG)

/**
 * @brief A multi-dimensional data container storing task data
 */
class PhysicalStore {
 public:
  /**
   * @brief Returns a read-only accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A read-only accessor to the store
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] AccessorRO<T, DIM> read_accessor() const;

  /**
   * @brief Returns a write-only accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A write-only accessor to the store
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] AccessorWO<T, DIM> write_accessor() const;

  /**
   * @brief Returns a read-write accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A read-write accessor to the store
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] AccessorRW<T, DIM> read_write_accessor() const;

  /**
   * @brief Returns a reduction accessor to the store for the entire domain.
   *
   * @tparam OP Reduction operator class.
   *
   * @tparam EXCLUSIVE Indicates whether reductions can be performed in exclusive mode. If
   * `EXCLUSIVE` is `false`, every reduction via the accessor is performed atomically.
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A reduction accessor to the store
   *
   * @see `Library::register_reduction_operator()`
   */
  template <typename OP,
            bool EXCLUSIVE,
            std::int32_t DIM,
            bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor() const;

  /**
   * @brief Returns a read-only accessor to the store for specific bounds.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed. The actual bounds for valid
   * access are determined by an intersection between the store's domain and the bounds.
   *
   * @return A read-only accessor to the store
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] AccessorRO<T, DIM> read_accessor(const Rect<DIM>& bounds) const;

  /**
   * @brief Returns a write-only accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed. The actual bounds for valid
   * access are determined by an intersection between the store's domain and the bounds.
   *
   * @return A write-only accessor to the store
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] AccessorWO<T, DIM> write_accessor(const Rect<DIM>& bounds) const;

  /**
   * @brief Returns a read-write accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed. The actual bounds for valid
   * access are determined by an intersection between the store's domain and the bounds.
   *
   * @return A read-write accessor to the store
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] AccessorRW<T, DIM> read_write_accessor(const Rect<DIM>& bounds) const;

  /**
   * @brief Returns a reduction accessor to the store for the entire domain.
   *
   * @tparam OP Reduction operator class.
   *
   * @tparam EXCLUSIVE Indicates whether reductions can be performed in exclusive mode. If
   * `EXCLUSIVE` is `false`, every reduction via the accessor is performed atomically.
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed. The actual bounds for valid
   * access are determined by an intersection between the store's domain and the bounds.
   *
   * @return A reduction accessor to the store
   *
   * @see `Library::register_reduction_operator()`
   */
  template <typename OP,
            bool EXCLUSIVE,
            std::int32_t DIM,
            bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(const Rect<DIM>& bounds) const;

  /**
   * @brief Returns the scalar value stored in the store.
   *
   * The requested type must match with the store's data type. If the store is not backed by
   * the future, the runtime will fail with an error message.
   *
   * @tparam VAL Type of the scalar value
   *
   * @return The scalar value stored in the store
   */
  template <typename VAL>
  [[nodiscard]] VAL scalar() const;

  /**
   * @brief Creates a \ref Buffer of specified extents for the unbound store.
   *
   * The returned \ref Buffer is always consistent with the mapping policy for the store. Can be
   * invoked multiple times unless `bind_buffer` is true.
   *
   * @param extents Extents of the \ref Buffer
   *
   * @param bind_buffer If the value is `true`, the created \ref Buffer will be bound to the
   * store upon return
   *
   * @return A \ref Buffer in which to write the output to.
   */
  template <typename T, std::int32_t DIM>
  [[nodiscard]] Buffer<T, DIM> create_output_buffer(const Point<DIM>& extents,
                                                    bool bind_buffer = false) const;

  /**
   * @brief Creates a `TaskLocalBuffer` of specified extents for the unbound store.
   *
   * The returned `TaskLocalBuffer` is always consistent with the mapping policy for the
   * store. Can be invoked multiple times unless `bind_buffer` is true.
   *
   * @param extents Extents of the `TaskLocalBuffer`
   * @param bind_buffer If the value is `true`, the created `TaskLocalBuffer` will be bound to the
   * store upon return.
   *
   * @return A `TaskLocalBuffer` in which to write the output to.
   */
  [[nodiscard]] TaskLocalBuffer create_output_buffer(const DomainPoint& extents,
                                                     bool bind_buffer = false) const;

  /**
   * @brief Binds a \ref Buffer to the store.
   *
   * Valid only when the store is unbound and has not yet been bound to another \ref
   * Buffer. The \ref Buffer must be consistent with the mapping policy for the store.
   * Recommend that the \ref Buffer be created by a `create_output_buffer()` call.
   *
   * @param buffer \ref Buffer to bind to the store
   *
   * @param extents Extents of the \ref Buffer. Passing extents smaller than the actual extents
   * of the \ref Buffer is legal; the runtime uses the passed extents as the extents of this
   * store.
   */
  template <typename T, std::int32_t DIM>
  void bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents) const;

  /**
   * @brief Binds a `TaskLocalBuffer` to the store.
   *
   * Valid only when the store is unbound and has not yet been bound to another
   * `TaskLocalBuffer`. The `TaskLocalBuffer` must be consistent with the mapping policy for
   * the store.  Recommend that the `TaskLocalBuffer` be created by a `create_output_buffer()`
   * call.
   *
   * Passing `extents` that are smaller than the actual extents of the `TaskLocalBuffer` is
   * legal; the runtime uses the passed extents as the extents of this store.
   *
   * If `check_type` is `true`, then `buffer` must have the same type as the `PhysicalStore`.
   *
   * @param buffer `TaskLocalBuffer` to bind to the store.
   * @param extents Extents of the `TaskLocalBuffer`.
   * @param check_type Whether to check the type of the buffer against the type of this store
   * for validity.
   *
   * @throw std::invalid_argument If the type of `buffer` is not compatible with the type of
   * the store (only thrown if `check_type` is `true`).
   */
  void bind_data(const TaskLocalBuffer& buffer,
                 const DomainPoint& extents,
                 bool check_type = false) const;

  /**
   * @brief Binds a 1D \ref Buffer of byte-size elements to the store in an untyped manner.
   *
   * Values in the \ref Buffer are reinterpreted based on the store's actual type. The \ref
   * Buffer must have enough bytes to be aligned on the store's element boundary. For example,
   * a 1D \ref Buffer of size 4 wouldn't be valid if the store had the int64 type, whereas it
   * would be if the store's element type is int32.
   *
   * Like the typed counterpart (i.e., `bind_data()`), the operation is legal only when the store is
   * unbound and has not yet been bound to another buffer. The memory in which the buffer is created
   * must be the same as the mapping decision of this store.
   *
   * Can be used only with 1D unbound stores.
   *
   * @param buffer \ref Buffer to bind to the store
   *
   * @param extents Extents of the buffer. Passing extents smaller than the actual extents of the
   * buffer is legal; the runtime uses the passed extents as the extents of this store. The size of
   * the buffer must be at least as big as `extents * type().size()`.
   *
   * @snippet unit/physical_store/create_unbound_store.cc Bind an untyped buffer to an unbound store
   */
  void bind_untyped_data(Buffer<std::int8_t, 1>& buffer, const Point<1>& extents) const;

  /**
   * @brief Makes the unbound store empty.
   *
   * Valid only when the store is unbound and has not yet been bound to another buffer.
   */
  void bind_empty_data() const;

  /**
   * @brief Returns the dimension of the store
   *
   * @return The store's dimension
   */
  [[nodiscard]] std::int32_t dim() const;

  /**
   * @brief Returns the type metadata of the store
   *
   * @return The store's `Type`
   */
  [[nodiscard]] Type type() const;

  /**
   * @brief Returns the type code of the store
   *
   * @return The store's type code
   */
  template <typename TYPE_CODE = Type::Code>
  [[nodiscard]] TYPE_CODE code() const;

  /**
   * @brief Returns the store's domain
   *
   * @return Store's domain
   */
  template <std::int32_t DIM>
  [[nodiscard]] Rect<DIM> shape() const;
  /**
   * @brief Returns the store's `Domain`
   *
   * @return Store's `Domain`
   */
  [[nodiscard]] Domain domain() const;

  /**
   * @brief Returns a raw pointer and strides to the allocation
   *
   * @return An `InlineAllocation` object holding a raw pointer and strides
   */
  [[nodiscard]] InlineAllocation get_inline_allocation() const;

  /**
   * @brief Returns the kind of memory where this `PhysicalStore` resides
   *
   * @return The memory kind
   *
   * @throw std::invalid_argument If this function is called on an unbound store
   */
  [[nodiscard]] mapping::StoreTarget target() const;

  /**
   * @brief Indicates whether the store can have a read accessor
   *
   * @return `true` if the store can have a read accessor, `false` otherwise
   */
  [[nodiscard]] bool is_readable() const;

  /**
   * @brief Indicates whether the store can have a write accessor
   *
   * @return `true` if the store can have a write accessor, `false` otherwise
   */
  [[nodiscard]] bool is_writable() const;

  /**
   * @brief Indicates whether the store can have a reduction accessor
   *
   * @return `true` if the store can have a reduction accessor, `false` otherwise
   */
  [[nodiscard]] bool is_reducible() const;

  /**
   * @brief Indicates whether the store is valid.
   *
   * A store passed to a task can be invalid only for reducer tasks for tree
   * reduction. Otherwise, if the store is invalid, it cannot be used in any data access.
   *
   * @return `true` if the store is valid, `false` otherwise
   */
  [[nodiscard]] bool valid() const;

  /**
   * @brief Indicates whether the store is transformed in any way.
   *
   * @return `true` if the store is transformed, `false` otherwise
   */
  [[nodiscard]] bool transformed() const;

  /**
   * @brief Indicates whether the store is backed by a future
   * (i.e., a container for scalar value)
   *
   * @return `true` if the store is backed by a future, `false` otherwise
   */
  [[nodiscard]] bool is_future() const;
  /**
   * @brief Indicates whether the store is an unbound store.
   *
   * The value DOES NOT indicate that the store has already assigned to a buffer; i.e., the store
   * may have been assigned to a buffer even when this function returns `true`.
   *
   * @return `true` if the store is an unbound store, `false` otherwise
   */
  [[nodiscard]] bool is_unbound_store() const;
  /**
   * @brief Indicates whether the store is partitioned.
   *
   * Tasks sometimes need to know whether a given `PhysicalStore` is partitioned, i.e., corresponds
   * to a subset of the (global) `LogicalStore` passed at the launch site. Unless the task
   * explicitly requests broadcasting on the `LogicalStore`, the partitioning decision on the store
   * is at the whim of the runtime. In this case, the task can use the `is_partitioned()` function
   * to retrieve that information.
   *
   * @return `true` if the store is partitioned, `false` otherwise
   */
  [[nodiscard]] bool is_partitioned() const;

  /**
   * @brief Constructs a store out of an array
   *
   * @throw std::invalid_argument If the array is nullable or has sub-arrays
   */
  // NOLINTNEXTLINE(google-explicit-constructor) very common pattern in cuPyNumeric
  PhysicalStore(const PhysicalArray& array);

  PhysicalStore() = LEGATE_DEFAULT_WHEN_CYTHON;

  explicit PhysicalStore(InternalSharedPtr<detail::PhysicalStore> impl,
                         std::optional<LogicalStore> owner = std::nullopt);

  [[nodiscard]] const SharedPtr<detail::PhysicalStore>& impl() const;

 private:
  void check_accessor_dimension_(std::int32_t dim) const;
  void check_buffer_dimension_(std::int32_t dim) const;
  void check_shape_dimension_(std::int32_t dim) const;
  void check_valid_binding_(bool bind_buffer) const;
  void check_write_access_() const;
  void check_reduction_access_() const;
  template <typename T>
  void check_accessor_type_() const;
  void check_accessor_type_(Type::Code code, std::size_t size_of_T) const;
  [[nodiscard]] Legion::DomainAffineTransform get_inverse_transform_() const;

  [[nodiscard]] std::pair<Legion::PhysicalRegion, Legion::FieldID> get_region_field_() const;
  [[nodiscard]] GlobalRedopID get_redop_id_() const;
  template <typename ACC, typename T, std::int32_t DIM>
  [[nodiscard]] ACC create_field_accessor_(const Rect<DIM>& bounds) const;
  template <typename ACC, typename T, std::int32_t DIM>
  [[nodiscard]] ACC create_reduction_accessor_(const Rect<DIM>& bounds) const;

  [[nodiscard]] bool is_read_only_future_() const;
  [[nodiscard]] std::size_t get_field_offset_() const;
  [[nodiscard]] const void* get_untyped_pointer_from_future_() const;
  [[nodiscard]] const Legion::Future& get_future_() const;
  [[nodiscard]] const Legion::UntypedDeferredValue& get_buffer_() const;

  [[nodiscard]] std::pair<Legion::OutputRegion, Legion::FieldID> get_output_field_() const;
  void update_num_elements_(std::size_t num_elements) const;

  [[noreturn]] static void throw_invalid_scalar_access_();

  SharedPtr<detail::PhysicalStore> impl_{};
  // This member exists purely to solve the temporary store problem. It is illegal for Physical
  // stores to outlive their LogicalStore counterparts, but it is pretty easy to get into a
  // situation where this happens. For example, you could do:
  //
  // auto phys = get_runtime()->create_store(...).get_physical_store();
  //
  // While this is illegal from the runtime perspective, we still want to make this "work" from
  // a user perspective, as it is very easy to get into. So we have this member. It's value is
  // immaterial (and should not be relied upon), and isn't exposed anywhere else.
  std::optional<LogicalStore> owner_{};
};

#undef LEGATE_TRUE_WHEN_DEBUG

/** @} */

}  // namespace legate

#include <legate/data/physical_store.inl>
