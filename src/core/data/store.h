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

#include "core/data/buffer.h"
#include "core/type/type_traits.h"
#include "core/utilities/dispatch.h"

/** @defgroup data Data abstractions and allocators
 */

/**
 * @file
 * @brief Class definition for legate::Store
 */

namespace legate::detail {
class Store;
}  // namespace legate::detail

namespace legate {

/**
 * @ingroup data
 *
 * @brief A multi-dimensional data container storing task data
 */
class Store {
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
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRO<T, DIM> read_accessor() const;
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
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorWO<T, DIM> write_accessor() const;
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
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRW<T, DIM> read_write_accessor() const;
  /**
   * @brief Returns a reduction accessor to the store for the entire domain.
   *
   * @tparam OP Reduction operator class. For details about reduction operators, See
   * Library::register_reduction_operator.
   *
   * @tparam EXCLUSIVE Indicates whether reductions can be performed in exclusive mode. If
   * `EXCLUSIVE` is `false`, every reduction via the accessor is performed atomically.
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A reduction accessor to the store
   */
  template <typename OP, bool EXCLUSIVE, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor() const;

 public:
  /**
   * @brief Returns a read-only accessor to the store for specific bounds.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed.
   * The actual bounds for valid access are determined by an intersection between
   * the store's domain and the bounds.
   *
   * @return A read-only accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRO<T, DIM> read_accessor(const Rect<DIM>& bounds) const;
  /**
   * @brief Returns a write-only accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed.
   * The actual bounds for valid access are determined by an intersection between
   * the store's domain and the bounds.
   *
   * @return A write-only accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorWO<T, DIM> write_accessor(const Rect<DIM>& bounds) const;
  /**
   * @brief Returns a read-write accessor to the store for the entire domain.
   *
   * @tparam T Element type
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @param bounds Domain within which accesses should be allowed.
   * The actual bounds for valid access are determined by an intersection between
   * the store's domain and the bounds.
   *
   * @return A read-write accessor to the store
   */
  template <typename T, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRW<T, DIM> read_write_accessor(const Rect<DIM>& bounds) const;
  /**
   * @brief Returns a reduction accessor to the store for the entire domain.
   *
   * @param bounds Domain within which accesses should be allowed.
   * The actual bounds for valid access are determined by an intersection between
   * the store's domain and the bounds.
   *
   * @tparam OP Reduction operator class. For details about reduction operators, See
   * Library::register_reduction_operator.
   *
   * @tparam EXCLUSIVE Indicates whether reductions can be performed in exclusive mode. If
   * `EXCLUSIVE` is `false`, every reduction via the accessor is performed atomically.
   *
   * @tparam DIM Number of dimensions
   *
   * @tparam VALIDATE_TYPE If `true` (default), validates type and number of dimensions
   *
   * @return A reduction accessor to the store
   */
  template <typename OP, bool EXCLUSIVE, int32_t DIM, bool VALIDATE_TYPE = true>
  AccessorRD<OP, EXCLUSIVE, DIM> reduce_accessor(const Rect<DIM>& bounds) const;

 public:
  /**
   * @brief Returns the scalar value stored in the store.
   *
   * The requested type must match with the store's data type. If the store is not
   * backed by the future, the runtime will fail with an error message.
   *
   * @tparam VAL Type of the scalar value
   *
   * @return The scalar value stored in the store
   */
  template <typename VAL>
  VAL scalar() const;

 public:
  /**
   * @brief Creates a buffer of specified extents for the unbound store. The returned
   * buffer is always consistent with the mapping policy for the store. Can be invoked
   * multiple times unless `bind_buffer` is true.
   *
   * @param extents Extents of the buffer
   *
   * @param bind_buffer If the value is true, the created buffer will be bound
   * to the store upon return
   *
   * @return A reduction accessor to the store
   */
  template <typename T, int32_t DIM>
  Buffer<T, DIM> create_output_buffer(const Point<DIM>& extents, bool bind_buffer = false);
  /**
   * @brief Binds a buffer to the store. Valid only when the store is unbound and
   * has not yet been bound to another buffer. The buffer must be consistent with
   * the mapping policy for the store. Recommend that the buffer be created by
   * a `create_output_buffer` call.
   *
   * @param buffer Buffer to bind to the store
   *
   * @param extents Extents of the buffer. Passing extents smaller than the actual
   * extents of the buffer is legal; the runtime uses the passed extents as the
   * extents of this store.
   *
   */
  template <typename T, int32_t DIM>
  void bind_data(Buffer<T, DIM>& buffer, const Point<DIM>& extents);
  /**
   * @brief Makes the unbound store empty. Valid only when the store is unbound and
   * has not yet been bound to another buffer.
   */
  void bind_empty_data();

 public:
  /**
   * @brief Returns the dimension of the store
   *
   * @return The store's dimension
   */
  int32_t dim() const;
  /**
   * @brief Returns the type metadata of the store
   *
   * @return The store's type metadata
   */
  Type type() const;
  /**
   * @brief Returns the type code of the store
   *
   * @return The store's type code
   */
  template <typename TYPE_CODE = Type::Code>
  TYPE_CODE code() const
  {
    return static_cast<TYPE_CODE>(type().code());
  }

 public:
  /**
   * @brief Returns the store's domain
   *
   * @return Store's domain
   */
  template <int32_t DIM>
  Rect<DIM> shape() const;
  /**
   * @brief Returns the store's domain in a dimension-erased domain type
   *
   * @return Store's domain in a dimension-erased domain type
   */
  Domain domain() const;

 public:
  /**
   * @brief Indicates whether the store can have a read accessor
   *
   * @return true The store can have a read accessor
   * @return false The store cannot have a read accesor
   */
  bool is_readable() const;
  /**
   * @brief Indicates whether the store can have a write accessor
   *
   * @return true The store can have a write accessor
   * @return false The store cannot have a write accesor
   */
  bool is_writable() const;
  /**
   * @brief Indicates whether the store can have a reduction accessor
   *
   * @return true The store can have a reduction accessor
   * @return false The store cannot have a reduction accesor
   */
  bool is_reducible() const;

 public:
  /**
   * @brief Indicates whether the store is valid. A store passed to a task can be invalid
   * only for reducer tasks for tree reduction.
   *
   * @return true The store is valid
   * @return false The store is invalid and cannot be used in any data access
   */
  bool valid() const;
  /**
   * @brief Indicates whether the store is transformed in any way.
   *
   * @return true The store is transformed
   * @return false The store is not transformed
   */
  bool transformed() const;

 public:
  /**
   * @brief Indicates whether the store is backed by a future
   * (i.e., a container for scalar value)
   *
   * @return true The store is backed by a future
   * @return false The store is backed by a region field
   */
  bool is_future() const;
  /**
   * @brief Indicates whether the store is an unbound store. The value DOES NOT indicate
   * that the store has already assigned to a buffer; i.e., the store may have been assigned
   * to a buffer even when this function returns `true`.
   *
   * @return true The store is an unbound store
   * @return false The store is a normal store
   */
  bool is_unbound_store() const;

 public:
  /**
   * @brief Releases all inline allocations of the store
   */
  void unmap();

 private:
  void check_accessor_dimension(const int32_t dim) const;
  void check_buffer_dimension(const int32_t dim) const;
  void check_shape_dimension(const int32_t dim) const;
  void check_valid_binding(bool bind_buffer) const;
  template <typename T>
  void check_accessor_type() const;
  Legion::DomainAffineTransform get_inverse_transform() const;

 private:
  void get_region_field(Legion::PhysicalRegion& pr, Legion::FieldID& fid) const;
  int32_t get_redop_id() const;
  template <typename ACC, int32_t DIM>
  ACC create_field_accessor(const Rect<DIM>& bounds) const;
  template <typename ACC, int32_t DIM>
  ACC create_reduction_accessor(const Rect<DIM>& bounds) const;

 private:
  bool is_read_only_future() const;
  Legion::Future get_future() const;
  Legion::UntypedDeferredValue get_buffer() const;

 private:
  void get_output_field(Legion::OutputRegion& out, Legion::FieldID& fid);
  void update_num_elements(size_t num_elements);

 public:
  Store();
  Store(std::shared_ptr<detail::Store> impl);
  std::shared_ptr<detail::Store> impl() const { return impl_; }

 public:
  Store(const Store&);
  Store& operator=(const Store&);
  Store(Store&&);
  Store& operator=(Store&&);

 public:
  ~Store();

 private:
  std::shared_ptr<detail::Store> impl_{nullptr};
};

}  // namespace legate

#include "core/data/store.inl"
