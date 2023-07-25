/* Copyright 2021-2022 NVIDIA Corporation
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

#include "core/data/scalar.h"
#include "core/data/detail/scalar.h"

namespace legate {

Scalar::Scalar(std::unique_ptr<detail::Scalar> impl) : impl_(impl.release()) {}

Scalar::Scalar(const Scalar& other) : impl_(new detail::Scalar(*other.impl_)) {}

Scalar::Scalar(Scalar&& other) : impl_(other.impl_) { other.impl_ = nullptr; }

Scalar::~Scalar() { delete impl_; }

Scalar::Scalar(Type type, const void* data, bool copy) : impl_(create_impl(type, data, copy)) {}

Scalar::Scalar(const std::string& string) : impl_(new detail::Scalar(string)) {}

Scalar& Scalar::operator=(const Scalar& other)
{
  *impl_ = *other.impl_;
  return *this;
}

Type Scalar::type() const { return Type(impl_->type()); }

size_t Scalar::size() const { return impl_->size(); }

const void* Scalar::ptr() const { return impl_->data(); }

/*static*/ detail::Scalar* Scalar::create_impl(Type type, const void* data, bool copy)
{
  return new detail::Scalar(std::move(type.impl()), data, copy);
}

}  // namespace legate
