/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <legate/data/detail/scalar.h>

#include <legate/data/scalar.h>
#include <legate/utilities/detail/traced_exception.h>

#include <fmt/format.h>

#include <stdexcept>
#include <utility>

namespace legate {

Scalar::Scalar(InternalSharedPtr<detail::Scalar> impl) : impl_{std::move(impl)} {}

Scalar::Scalar(std::unique_ptr<detail::Scalar> impl) : impl_{std::move(impl)} {}

Scalar::Scalar() : Scalar{create_impl_(null_type(), nullptr, false), private_tag{}} {}

Scalar::Scalar(const Type& type, const void* data, bool copy)
  : Scalar{create_impl_(type, data, copy), private_tag{}}
{
}

Scalar::Scalar(std::string_view string)
  : impl_{legate::make_shared<detail::Scalar>(std::move(string))}
{
}

Type Scalar::type() const { return Type{impl_->type()}; }

std::size_t Scalar::size() const { return impl_->size(); }

template <>
std::string_view Scalar::value() const
{
  if (type().code() != Type::Code::STRING) {
    throw detail::TracedException<std::invalid_argument>{"Type of the scalar is not string"};
  }

  const void* data  = ptr();
  auto len          = *static_cast<const std::uint32_t*>(data);
  const auto* begin = static_cast<const char*>(data) + sizeof(len);
  return {begin, len};
}

template <>
std::string Scalar::value() const
{
  return std::string{this->value<std::string_view>()};
}

template <>
Legion::DomainPoint Scalar::value<Legion::DomainPoint>() const
{
  Legion::DomainPoint result;
  const auto span = values<std::int64_t>();

  result.dim = static_cast<decltype(result.dim)>(span.size());
  for (auto idx = 0; idx < result.dim; ++idx) {
    result[idx] = span[idx];
  }
  return result;
}

const void* Scalar::ptr() const { return impl_->data(); }

/*static*/ detail::Scalar* Scalar::checked_create_impl_(const Type& type,
                                                        const void* data,
                                                        bool copy,
                                                        std::size_t size)
{
  if (type.code() == Type::Code::NIL) {
    throw detail::TracedException<std::invalid_argument>{"Null type cannot be used"};
  }
  if (type.size() != size) {
    throw detail::TracedException<std::invalid_argument>{
      "Size of the value doesn't match with the type"};
  }

  return create_impl_(type, data, copy);
}

/*static*/ detail::Scalar* Scalar::create_impl_(const Type& type, const void* data, bool copy)
{
  LEGATE_CHECK(data || !copy);
  return new detail::Scalar{type.impl(), data, copy};
}

/*static*/ void Scalar::throw_invalid_size_exception_(std::size_t type_size, std::size_t size_of_T)
{
  throw detail::TracedException<std::invalid_argument>{fmt::format(
    "Size of the scalar is {}, but the requested type has size {}", type_size, size_of_T)};
}

/*static*/ void Scalar::throw_invalid_type_conversion_exception_(std::string_view from,
                                                                 std::string_view to)
{
  throw detail::TracedException<std::invalid_argument>{
    fmt::format("{} cannot be casted to {}", from, to)};
}

/*static*/ void Scalar::throw_invalid_span_conversion_exception_(Type::Code code,
                                                                 std::string_view kind,
                                                                 std::size_t expected,
                                                                 std::size_t actual)
{
  throw detail::TracedException<std::invalid_argument>{fmt::format(
    "{} scalar can only be converted into a span of a type whose {} is {} bytes (have {})",
    code,
    kind,
    expected,
    actual)};
}

Scalar::Scalar(detail::Scalar* impl, private_tag) : impl_{impl} {}

}  // namespace legate
