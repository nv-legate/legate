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

#include "core/data/detail/scalar.h"

#include "core/data/scalar.h"

#include <stdexcept>
#include <utility>

namespace legate {

Scalar::Scalar(const Scalar& other) : impl_{new detail::Scalar{*other.impl_}} {}

Scalar::Scalar(Scalar&& other) noexcept : impl_{std::exchange(other.impl_, nullptr)} {}

Scalar::Scalar() : impl_(create_impl(null_type(), nullptr, false)) {}

Scalar::~Scalar() { delete impl_; }

Scalar::Scalar(const Type& type, const void* data, bool copy) : impl_{create_impl(type, data, copy)}
{
}

Scalar::Scalar(const std::string& string) : impl_{new detail::Scalar{string}} {}

Scalar& Scalar::operator=(const Scalar& other)
{
  if (this != &other) {
    *impl_ = *other.impl_;
  }
  return *this;
}

Type Scalar::type() const { return Type{impl_->type()}; }

size_t Scalar::size() const { return impl_->size(); }

const void* Scalar::ptr() const { return impl_->data(); }

/*static*/ detail::Scalar* Scalar::checked_create_impl(const Type& type,
                                                       const void* data,
                                                       bool copy,
                                                       size_t size)
{
  if (type.code() == Type::Code::NIL) {
    throw std::invalid_argument{"Null type cannot be used"};
  }
  if (type.size() != size) {
    throw std::invalid_argument{"Size of the value doesn't match with the type"};
  }

  return create_impl(type, data, copy);
}

/*static*/ detail::Scalar* Scalar::create_impl(const Type& type, const void* data, bool copy)
{
  return new detail::Scalar{type.impl(), data, copy};
}

Scalar null() { return {}; }

}  // namespace legate
