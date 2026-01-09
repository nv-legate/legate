/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <legate/utilities/detail/dlpack/dlpack.h>
#include <legate/utilities/detail/doxygen.h>
#include <legate/utilities/detail/mdspan/reduction_accessor.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/mdspan.h>
#include <legate/utilities/shared_ptr.h>

#include <cuda/std/mdspan>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

/**
 * @file
 * @brief Class definition for legate::PhysicalStore
 */

struct CUstream_st;

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
class LEGATE_EXPORT PhysicalStore {
 public:
  template <typename T,
            std::uint32_t DIM,
            typename AccessorPolicy = ::cuda::std::default_accessor<T>>
  using mdspan_type = ::cuda::std::
    mdspan<T, ::cuda::std::dextents<coord_t, DIM>, ::cuda::std::layout_stride, AccessorPolicy>;

  /**
   * @brief Returns a read-only mdspan to the store over its entire domain. Equivalent to
   * `span_read_accessor<T, DIM>(shape<DIM>())`.
   *
   * @note This API is experimental. It will eventually replace the Legion accessor interface,
   * but we are seeking user feedback on it before such time. If you encounter issues, and/or
   * have suggestions for improvements, please file a bug at
   * https://github.com/nv-legate/legate/issues.
   *
   * `elem_size` should not normally need to be passed. It is, however, necessary for
   * type-punning when the size of the stored type and viewed type differ. For example, a store
   * might refer to binary data where each element is of size 10, but we wish to view it as an
   * mspan of `std::byte`s (which would have size = 1). In this case:
   *
   * #. Pass the viewed type (e.g. `std::byte`) as the template parameter `T`.
   * #. Pass `VALIDATE_TYPE = false` (to avoid warnings about type mismatch).
   * #. Pass `type().size()` as `elem_size`.
   *
   * @tparam T The element type of the mdspan.
   * @tparam DIM The rank of the mdspan.
   * @tparam VALIDATE_TYPE If `true` (default), checks that the type and rank of the mdspan
   * match that of the `PhysicalStore`.
   *
   * @param elem_size The size (in bytes) of each element.
   *
   * @return The read-only mdspan accessor.
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] mdspan_type<const T, DIM> span_read_accessor(
    std::size_t elem_size = sizeof(T)) const;

  /**
   * @brief Returns a write-only mdspan to the store over its entire domain. Equivalent to
   * `span_write_accessor<T, DIM>(shape<DIM>())`.
   *
   * @note This API is experimental. It will eventually replace the Legion accessor interface,
   * but we are seeking user feedback on it before such time. If you encounter issues, and/or
   * have suggestions for improvements, please file a bug at
   * https://github.com/nv-legate/legate/issues.
   *
   * The user may read from a write-only accessor, but must write to the read-from location
   * first, otherwise the returned values are undefined:
   *
   * @code{.cpp}
   * auto acc = store.span_write_accessor<float, 2>();
   *
   * v = acc(0, 0); // Note: undefined value
   * acc(0, 0) = 42.0;
   * v = acc(0, 0); // OK, value will be 42.0
   * @endcode
   *
   * `elem_size` should not normally need to be passed. It is, however, necessary for
   * type-punning when the size of the stored type and viewed type differ. For example, a store
   * might refer to binary data where each element is of size 10, but we wish to view it as an
   * mspan of `std::byte`s (which would have size = 1). In this case:
   *
   * #. Pass the viewed type (e.g. `std::byte`) as the template parameter `T`.
   * #. Pass `VALIDATE_TYPE = false` (to avoid warnings about type mismatch).
   * #. Pass `type().size()` as `elem_size`.
   *
   * @tparam T The element type of the mdspan.
   * @tparam DIM The rank of the mdspan.
   * @tparam VALIDATE_TYPE If `true` (default on debug builds), checks that the type and rank
   * of the mdspan match that of the `PhysicalStore`.
   *
   * @param elem_size The size (in bytes) of each element.
   *
   * @return The mdspan accessor.
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] mdspan_type<T, DIM> span_write_accessor(std::size_t elem_size = sizeof(T));

  /**
   * @brief Returns a read-write mdspan to the store over its entire domain. Equivalent to
   * `span_read_write_accessor<T, DIM>(shape<DIM>())`.
   *
   * @note This API is experimental. It will eventually replace the Legion accessor interface,
   * but we are seeking user feedback on it before such time. If you encounter issues, and/or
   * have suggestions for improvements, please file a bug at
   * https://github.com/nv-legate/legate/issues.
   *
   * `elem_size` should not normally need to be passed. It is, however, necessary for
   * type-punning when the size of the stored type and viewed type differ. For example, a store
   * might refer to binary data where each element is of size 10, but we wish to view it as an
   * mspan of `std::byte`s (which would have size = 1). In this case:
   *
   * #. Pass the viewed type (e.g. `std::byte`) as the template parameter `T`.
   * #. Pass `VALIDATE_TYPE = false` (to avoid warnings about type mismatch).
   * #. Pass `type().size()` as `elem_size`.
   *
   * @tparam T The element type of the mdspan.
   * @tparam DIM The rank of the mdspan.
   * @tparam VALIDATE_TYPE If `true` (default on debug builds), checks that the type and rank
   * of the mdspan match that of the `PhysicalStore`.
   *
   * @param elem_size The size (in bytes) of each element.
   *
   * @return The mdspan accessor.
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] mdspan_type<T, DIM> span_read_write_accessor(std::size_t elem_size = sizeof(T));

  /**
   * @brief Returns a reduction mdspan to the store over its entire domain. Equivalent to
   * `span_reduce_accessor<Redop, EXCLUSIVE, DIM>(shape<DIM>())`.
   *
   * @note This API is experimental. It will eventually replace the Legion accessor interface,
   * but we are seeking user feedback on it before such time. If you encounter issues, and/or
   * have suggestions for improvements, please file a bug at
   * https://github.com/nv-legate/legate/issues.
   *
   * `elem_size` should not normally need to be passed. It is, however, necessary for
   * type-punning when the size of the stored type and viewed type differ. For example, a store
   * might refer to binary data where each element is of size 10, but we wish to view it as an
   * mspan of `std::byte`s (which would have size = 1). In this case:
   *
   * #. Pass the viewed type (e.g. `std::byte`) as the template parameter `T`.
   * #. Pass `VALIDATE_TYPE = false` (to avoid warnings about type mismatch).
   * #. Pass `type().size()` as `elem_size`.
   *
   * @tparam Redop The reduction operator (e.g. `SumReduction`).
   * @tparam EXCLUSIVE Whether the reduction accessor has exclusive access to the buffer.
   * @tparam DIM The rank of the mdspan.
   * @tparam VALIDATE_TYPE If `true` (default on debug builds), checks that the type and rank
   * of the mdspan match that of the `PhysicalStore`.
   *
   * @param elem_size The size (in bytes) of each element.
   *
   * @return The mdspan accessor.
   */
  template <typename Redop,
            bool EXCLUSIVE,
            std::int32_t DIM,
            bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] mdspan_type<typename Redop::LHS, DIM, detail::ReductionAccessor<Redop, EXCLUSIVE>>
  span_reduce_accessor(std::size_t elem_size = sizeof(typename Redop::LHS));

  /**
   * @brief Returns a read-only mdspan to the store over the selected domain.
   *
   * @note This API is experimental. It will eventually replace the Legion accessor interface,
   * but we are seeking user feedback on it before such time. If you encounter issues, and/or
   * have suggestions for improvements, please file a bug at
   * https://github.com/nv-legate/legate/issues.
   *
   * `elem_size` should not normally need to be passed. It is, however, necessary for
   * type-punning when the size of the stored type and viewed type differ. For example, a store
   * might refer to binary data where each element is of size 10, but we wish to view it as an
   * mspan of `std::byte`s (which would have size = 1). In this case:
   *
   * #. Pass the viewed type (e.g. `std::byte`) as the template parameter `T`.
   * #. Pass `VALIDATE_TYPE = false` (to avoid warnings about type mismatch).
   * #. Pass `type().size()` as `elem_size`.
   *
   * @note If `bounds` is empty then the strides of the returned `mdspan` will be all 0 instead
   * of what it might normally be. The object is still perfectly usable as normal but the
   * strides will not be correct.
   *
   * @tparam T The element type of the mdspan.
   * @tparam DIM The rank of the mdspan.
   * @tparam VALIDATE_TYPE If `true` (default on debug builds), checks that the type and rank
   * of the mdspan match that of the `PhysicalStore`.
   *
   * @param bounds The (sub-)domain over which to access the store.
   * @param elem_size The size (in bytes) of each element.
   *
   * @return The mdspan accessor.
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] mdspan_type<const T, DIM> span_read_accessor(
    const Rect<DIM>& bounds, std::size_t elem_size = sizeof(T)) const;

  /**
   * @brief Returns a write-only mdspan to the store over the selected domain.
   *
   * @note This API is experimental. It will eventually replace the Legion accessor interface,
   * but we are seeking user feedback on it before such time. If you encounter issues, and/or
   * have suggestions for improvements, please file a bug at
   * https://github.com/nv-legate/legate/issues.
   *
   * The user may read from a write-only accessor, but must write to the read-from location
   * first, otherwise the returned values are undefined:
   *
   * @code{.cpp}
   * auto acc = store.span_write_accessor<float, 2>(bounds);
   *
   * v = acc(0, 0); // Note: undefined value
   * acc(0, 0) = 42.0;
   * v = acc(0, 0); // OK, value will be 42.0
   * @endcode
   *
   * @note If `bounds` is empty then the strides of the returned `mdspan` will be all 0 instead
   * of what it might normally be. The object is still perfectly usable as normal but the
   * strides will not be correct.
   *
   * `elem_size` should not normally need to be passed. It is, however, necessary for
   * type-punning when the size of the stored type and viewed type differ. For example, a store
   * might refer to binary data where each element is of size 10, but we wish to view it as an
   * mspan of `std::byte`s (which would have size = 1). In this case:
   *
   * #. Pass the viewed type (e.g. `std::byte`) as the template parameter `T`.
   * #. Pass `VALIDATE_TYPE = false` (to avoid warnings about type mismatch).
   * #. Pass `type().size()` as `elem_size`.
   *
   * @tparam T The element type of the mdspan.
   * @tparam DIM The rank of the mdspan.
   * @tparam VALIDATE_TYPE If `true` (default on debug builds), checks that the type and rank
   * of the mdspan match that of the `PhysicalStore`.
   *
   * @param bounds The (sub-)domain over which to access the store.
   * @param elem_size The size (in bytes) of each element.
   *
   * @return The mdspan accessor.
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] mdspan_type<T, DIM> span_write_accessor(const Rect<DIM>& bounds,
                                                        std::size_t elem_size = sizeof(T));

  /**
   * @brief Returns a read-write mdspan to the store over the selected domain.
   *
   * @note This API is experimental. It will eventually replace the Legion accessor interface,
   * but we are seeking user feedback on it before such time. If you encounter issues, and/or
   * have suggestions for improvements, please file a bug at
   * https://github.com/nv-legate/legate/issues.
   *
   * @note If `bounds` is empty then the strides of the returned `mdspan` will be all 0 instead
   * of what it might normally be. The object is still perfectly usable as normal but the
   * strides will not be correct.
   *
   * `elem_size` should not normally need to be passed. It is, however, necessary for
   * type-punning when the size of the stored type and viewed type differ. For example, a store
   * might refer to binary data where each element is of size 10, but we wish to view it as an
   * mspan of `std::byte`s (which would have size = 1). In this case:
   *
   * #. Pass the viewed type (e.g. `std::byte`) as the template parameter `T`.
   * #. Pass `VALIDATE_TYPE = false` (to avoid warnings about type mismatch).
   * #. Pass `type().size()` as `elem_size`.
   *
   * @tparam T The element type of the mdspan.
   * @tparam DIM The rank of the mdspan.
   * @tparam VALIDATE_TYPE If `true` (default on debug builds), checks that the type and rank
   * of the mdspan match that of the `PhysicalStore`.
   *
   * @param bounds The (sub-)domain over which to access the store.
   * @param elem_size The size (in bytes) of each element.
   *
   * @return The mdspan accessor.
   */
  template <typename T, std::int32_t DIM, bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] mdspan_type<T, DIM> span_read_write_accessor(const Rect<DIM>& bounds,
                                                             std::size_t elem_size = sizeof(T));

  /**
   * @brief Returns a reduction mdspan to the store over the selected domain.
   *
   * @note This API is experimental. It will eventually replace the Legion accessor interface,
   * but we are seeking user feedback on it before such time. If you encounter issues, and/or
   * have suggestions for improvements, please file a bug at
   * https://github.com/nv-legate/legate/issues.
   *
   * @note If `bounds` is empty then the strides of the returned `mdspan` will be all 0 instead
   * of what it might normally be. The object is still perfectly usable as normal but the
   * strides will not be correct.
   *
   * `elem_size` should not normally need to be passed. It is, however, necessary for
   * type-punning when the size of the stored type and viewed type differ. For example, a store
   * might refer to binary data where each element is of size 10, but we wish to view it as an
   * mspan of `std::byte`s (which would have size = 1). In this case:
   *
   * #. Pass the viewed type (e.g. `std::byte`) as the template parameter `T`.
   * #. Pass `VALIDATE_TYPE = false` (to avoid warnings about type mismatch).
   * #. Pass `type().size()` as `elem_size`.
   *
   * @tparam Redop The reduction operator (e.g. `SumReduction`).
   * @tparam EXCLUSIVE Whether the reduction accessor has exclusive access to the buffer.
   * @tparam DIM The rank of the mdspan.
   * @tparam VALIDATE_TYPE If `true` (default on debug builds), checks that the type and rank
   * of the mdspan match that of the `PhysicalStore`.
   *
   * @param bounds The (sub-)domain over which to access the store.
   * @param elem_size The size (in bytes) of each element.
   *
   * @return The mdspan accessor.
   */
  template <typename Redop,
            bool EXCLUSIVE,
            std::int32_t DIM,
            bool VALIDATE_TYPE = LEGATE_TRUE_WHEN_DEBUG>
  [[nodiscard]] mdspan_type<typename Redop::LHS, DIM, detail::ReductionAccessor<Redop, EXCLUSIVE>>
  span_reduce_accessor(const Rect<DIM>& bounds,
                       std::size_t elem_size = sizeof(typename Redop::LHS));

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
   * @brief Export this store into the DLPack format.
   *
   * The value of `copy` has the following semantics:
   *
   * - `true`: Legate *must* copy the data. If Legate fails to copy the data, for any reason,
   *   an exception is thrown.
   * - `false`: Legate must *never* copy the data. If the store cannot be exported without a
   *   copy, then an exception is thrown.
   * - `std::nullopt`: Legate may copy the data if it is deemed necessary. Currently, this is
   *   never the case, and Legate will always provide a view.
   *
   * In any case, if a copy is made, the `DLManagedTensorVersioned::flags` member will have the
   * `DLPACK_FLAG_BITMASK_IS_COPIED` bit set.
   *
   * The `std::unique_ptr` returned by this routine will automatically call the deleter of the
   * DLPack tensor in its destructor.
   *
   * @param copy Whether to copy the underlying data or not.
   * @param stream A stream on which the data must be coherent after this routine returns.
   *
   * @return The DLPack managed tensor.
   */
  [[nodiscard]] std::unique_ptr<DLManagedTensorVersioned, void (*)(DLManagedTensorVersioned*)>
  to_dlpack(std::optional<bool> copy           = std::nullopt,
            std::optional<CUstream_st*> stream = std::nullopt) const;

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
  void check_accessor_store_backing_() const;
  void check_buffer_dimension_(std::int32_t dim) const;
  void check_shape_dimension_(std::int32_t dim) const;
  void check_valid_binding_(bool bind_buffer) const;
  void check_write_access_() const;
  void check_reduction_access_() const;
  template <typename T>
  void check_accessor_type_() const;
  void check_accessor_type_(Type::Code code, std::size_t size_of_T) const;
  void check_scalar_store_() const;
  void check_unbound_store_() const;
  [[nodiscard]] Legion::DomainAffineTransform get_inverse_transform_() const;

  [[nodiscard]] std::pair<Legion::PhysicalRegion, Legion::FieldID> get_region_field_() const;
  [[nodiscard]] GlobalRedopID get_redop_id_() const;

  /**
   * @brief Create a standard read/read-write/write-only field accessor over the underlying
   * data of the store.
   *
   * Type validation is usually enabled in debug-mode, and disabled in release mode. But it may
   * be useful to disable it wholesale, for example when punning the underlying buffer as some
   * other data-type. In this case, the user must take care to pass the right field sizes and
   * shapes to the accessor, but is otherwise free to recast the data as they see fit.
   *
   * @tparam ACC The accessor type to create.
   * @tparam T The value type of the accessor (since Legion accessors don't expose this
   * directly).
   * @tparam DIM The dimension of the accessor (and therefore the shape).
   *
   * @param bounds The bounds over which the accessor should provide access. Must be within the
   * bounds of the store's shape.
   * @param validate_type Whether to perform type size verification.
   *
   * @return The accessor.
   */
  template <typename ACC, typename T, std::int32_t DIM>
  [[nodiscard]] ACC create_field_accessor_(const Rect<DIM>& bounds, bool validate_type) const;

  /**
   * @brief Create a reduction field accessor over the underlying data.
   *
   * Type validation is usually enabled in debug-mode, and disabled in release mode. But it may
   * be useful to disable it wholesale, for example when punning the underlying buffer as some
   * other data-type. In this case, the user must take care to pass the right field sizes and
   * shapes to the accessor, but is otherwise free to recast the data as they see fit.
   *
   * @tparam ACC The accessor type to create.
   * @tparam T The value type of the accessor (since Legion accessors don't expose this
   * directly).
   * @tparam DIM The dimension of the accessor (and therefore the shape).
   *
   * @param bounds The bounds over which the accessor should provide access. Must be within the
   * bounds of the store's shape.
   * @param validate_type Whether to perform type size verification.
   *
   * @return The accessor.
   */
  template <typename ACC, typename T, std::int32_t DIM>
  [[nodiscard]] ACC create_reduction_accessor_(const Rect<DIM>& bounds, bool validate_type) const;

  [[nodiscard]] bool is_read_only_future_() const;
  [[nodiscard]] std::size_t get_field_offset_() const;
  [[nodiscard]] const void* get_untyped_pointer_from_future_() const;
  [[nodiscard]] const Legion::Future& get_future_() const;
  [[nodiscard]] const Legion::UntypedDeferredValue& get_buffer_() const;

  [[nodiscard]] std::pair<Legion::OutputRegion, Legion::FieldID> get_output_field_() const;
  void update_num_elements_(std::size_t num_elements) const;

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
