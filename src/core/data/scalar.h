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

#include "core/type/type_traits.h"
#include "core/utilities/span.h"
#include "core/utilities/tuple.h"
#include "core/utilities/typedefs.h"

/**
 * @file
 * @brief Class definition for legate::Scalar
 */

namespace legate::detail {
class Scalar;
}  // namespace legate::detail

namespace legate {

class AutoTask;
class ManualTask;
class Runtime;

/**
 * @ingroup data
 * @brief A type-erased container for scalars
 *
 * A Scalar can be owned or shared, depending on whether it owns the backing allocation:
 * If a `Scalar` is shared, it does not own the allocation and any of its copies are also
 * shared. If a `Scalar` is owned, it owns the backing allocation and releases it upon
 * destruction. Any copy of an owned `Scalar` is owned as well.
 */
class Scalar {
 public:
  Scalar(std::unique_ptr<detail::Scalar> impl);
  Scalar(const Scalar& other);
  Scalar(Scalar&& other);
  ~Scalar();

 public:
  /**
   * @brief Creates a null scalar
   */
  Scalar();
  /**
   * @brief Creates a shared `Scalar` with an existing allocation. The caller is responsible
   * for passing in a sufficiently big allocation.
   *
   * @param type Type of the scalar
   * @param data Allocation containing the data.
   * @param copy If true, the scalar copies the data stored in the allocation and becomes owned.
   */
  Scalar(Type type, const void* data, bool copy = false);
  /**
   * @brief Creates an owned scalar from a scalar value
   *
   * @tparam T The scalar type to wrap
   *
   * @param value A scalar value to create a `Scalar` with
   */
  template <typename T>
  Scalar(T value);
  /**
   * @brief Creates an owned scalar of a specified type from a scalar value
   *
   * @tparam T The scalar type to wrap
   *
   * @param type The type of the scalar
   * @param value A scalar value to create a `Scalar` with
   */
  template <typename T>
  Scalar(T value, Type type);
  /**
   * @brief Creates an owned scalar from a string. The value from the
   * original string will be copied.
   *
   * @param string A string to create a `Scalar` with
   */
  Scalar(const std::string& string);
  /**
   * @brief Creates an owned scalar from a vector of scalars. The values in the input vector
   * will be copied.
   *
   * @param values Values to create a scalar with in a vector
   */
  template <typename T>
  Scalar(const std::vector<T>& values);
  /**
   * @brief Creates an owned scalar from a tuple of scalars. The values in the input tuple
   * will be copied.
   *
   * @param values Values to create a scalar with in a tuple
   */
  template <typename T>
  Scalar(const tuple<T>& values);

 public:
  /**
   * @brief Creates a point scalar
   *
   * @param point A point from which the scalar should be constructed
   */
  template <int32_t DIM>
  Scalar(const Point<DIM>& point);
  /**
   * @brief Creates a rect scalar
   *
   * @param rect A rect from which the scalar should be constructed
   */
  template <int32_t DIM>
  Scalar(const Rect<DIM>& rect);

 public:
  Scalar& operator=(const Scalar& other);

 public:
  /**
   * @brief Returns the data type of the scalar
   *
   * @return Data type
   */
  Type type() const;
  /**
   * @brief Returns the size of allocation for the `Scalar`.
   *
   * @return The size of allocation
   */
  size_t size() const;

 public:
  /**
   * @brief Returns a copy of the value stored in this `Scalar`.
   *
   * @tparam VAL Type of the value to unwrap
   *
   * @return A copy of the value stored in this `Scalar`
   *
   * @throw std::invalid_argument If one of the following cases is encountered:
   *
   * 1) size of the scalar does not match with size of `VAL`,
   * 2) the scalar holds a string but `VAL` isn't `std:string` or `std:string_view`, or
   * 3) the inverse; i.e.,  `VAL` is `std:string` or `std:string_view` but the scalar's type
   * isn't string
   */
  template <typename VAL>
  VAL value() const;
  /**
   * @brief Returns values stored in the `Scalar`. If the `Scalar` does not have a fixed array type,
   * a unit span will be returned.
   *
   * @return Values stored in the `Scalar`
   *
   * @throw std::invalid_argument If one of the following cases is encountered:
   *
   * 1) the scalar has a fixed array type whose elemenet type has a different size from `VAL`,
   * 2) the scalar holds a string and size of `VAL` isn't 1 byte,
   * 3) the scalar's type isn't a fixed array type and the size is different from size of `VAL`
   */
  template <typename VAL>
  Span<const VAL> values() const;
  /**
   * @brief Returns a raw pointer to the backing allocation
   *
   * @return A raw pointer to the `Scalar`'s data
   */
  const void* ptr() const;

 public:
  detail::Scalar* impl() const { return impl_; }

 private:
  static detail::Scalar* create_impl(Type type, const void* data, bool copy);

  struct private_tag {};

  template <typename T>
  Scalar(T value, private_tag);

  friend class AutoTask;
  friend class ManualTask;
  friend class Runtime;
  detail::Scalar* impl_{nullptr};
};

Scalar null();

}  // namespace legate

#include "core/data/scalar.inl"
