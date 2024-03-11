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

#include "core/utilities/shared_ptr.h"

namespace legate {

template <typename T>
void SharedPtr<T>::reference_() noexcept
{
  internal_ptr({}).user_reference_({});
}

template <typename T>
void SharedPtr<T>::dereference_() noexcept
{
  internal_ptr({}).user_dereference_({});
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(copy_tag, const InternalSharedPtr<U>& other) noexcept : ptr_{other}
{
  reference_();
}

template <typename T>
template <typename U>
SharedPtr<T>::SharedPtr(move_tag, InternalSharedPtr<U>&& other, bool from_internal_ptr) noexcept
  : ptr_{std::move(other)}
{
  // Only update refcount if we are constructing from a bare InternalSharedPointer, since
  // the previous owning SharedPtr gives ownership entirely
  if (from_internal_ptr) {
    reference_();
  }
}

// ==========================================================================================

template <typename T>
SharedPtr<T>::SharedPtr(std::nullptr_t) noexcept
{
}

template <typename T>
template <typename U, typename Deleter, typename Alloc, typename SFINAE>
SharedPtr<T>::SharedPtr(U* ptr, Deleter deleter, Alloc allocator)
  : ptr_{ptr, std::move(deleter), std::move(allocator)}
{
  reference_();
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>::SharedPtr(U* ptr) : SharedPtr{ptr, detail::shared_ptr_default_delete<T, U>{}}
{
}

template <typename T>
SharedPtr<T>::SharedPtr(const SharedPtr& other) noexcept
  : SharedPtr{copy_tag{}, other.internal_ptr({})}
{
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr& other) noexcept
{
  SharedPtr{other}.swap(*this);
  return *this;
}

template <typename T>
SharedPtr<T>::SharedPtr(SharedPtr&& other) noexcept
  : SharedPtr{move_tag{}, std::move(other.internal_ptr({})), false}
{
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr&& other) noexcept
{
  SharedPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>::SharedPtr(const SharedPtr<U>& other) noexcept
  : SharedPtr{copy_tag{}, other.internal_ptr({})}
{
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr<U>& other) noexcept
{
  SharedPtr{other}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>::SharedPtr(SharedPtr<U>&& other) noexcept
  : SharedPtr{move_tag{}, std::move(other.internal_ptr({})), false}
{
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr<U>&& other) noexcept
{
  SharedPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
SharedPtr<T>::SharedPtr(const InternalSharedPtr<element_type>& other) noexcept
  : SharedPtr{copy_tag{}, other}
{
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(const InternalSharedPtr<element_type>& other) noexcept
{
  SharedPtr{other}.swap(*this);
  return *this;
}

template <typename T>
SharedPtr<T>::SharedPtr(InternalSharedPtr<element_type>&& other) noexcept
  : SharedPtr{move_tag{}, std::move(other), true}
{
}

template <typename T>
SharedPtr<T>& SharedPtr<T>::operator=(InternalSharedPtr<element_type>&& other) noexcept
{
  SharedPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>::SharedPtr(const InternalSharedPtr<U>& other) noexcept : SharedPtr{copy_tag{}, other}
{
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>& SharedPtr<T>::operator=(const InternalSharedPtr<U>& other) noexcept
{
  SharedPtr{other}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>::SharedPtr(InternalSharedPtr<U>&& other) noexcept
  : SharedPtr{move_tag{}, std::move(other), true}
{
}

template <typename T>
template <typename U, typename SFINAE>
SharedPtr<T>& SharedPtr<T>::operator=(InternalSharedPtr<U>&& other) noexcept
{
  SharedPtr{std::move(other)}.swap(*this);
  return *this;
}

template <typename T>
template <typename U, typename D, typename SFINAE>
SharedPtr<T>::SharedPtr(std::unique_ptr<U, D>&& ptr) : ptr_{std::move(ptr)}
{
  reference_();
}

template <typename T>
template <typename U, typename D, typename SFINAE>
SharedPtr<T>& SharedPtr<T>::operator=(std::unique_ptr<U, D>&& ptr)
{
  SharedPtr{std::move(ptr)}.swap(*this);
  return *this;
}

template <typename T>
SharedPtr<T>::~SharedPtr() noexcept
{
  dereference_();
}

// ==========================================================================================

template <typename T>
void SharedPtr<T>::swap(SharedPtr& other) noexcept
{
  internal_ptr({}).swap(other.internal_ptr({}));
}

// friend function
template <typename T>
void swap(SharedPtr<T>& lhs, SharedPtr<T>& rhs) noexcept
{
  lhs.swap(rhs);
}

template <typename T>
void SharedPtr<T>::reset() noexcept
{
  reset(nullptr);
}

template <typename T>
void SharedPtr<T>::reset(std::nullptr_t) noexcept
{
  SharedPtr<T>{nullptr}.swap(*this);
}

template <typename T>
template <typename U, typename D, typename A, typename SFINAE>
void SharedPtr<T>::reset(U* ptr, D deleter, A allocator)
{
  // cannot call ptr_.reset() since we may need to bump the user and strong reference counts
  SharedPtr<T>{ptr, std::move(deleter), std::move(allocator)}.swap(*this);
}

// ==========================================================================================

template <typename T>
typename SharedPtr<T>::element_type& SharedPtr<T>::operator[](std::ptrdiff_t idx) noexcept
{
  return internal_ptr({})[idx];
}

template <typename T>
const typename SharedPtr<T>::element_type& SharedPtr<T>::operator[](
  std::ptrdiff_t idx) const noexcept
{
  return internal_ptr({})[idx];
}

template <typename T>
typename SharedPtr<T>::element_type* SharedPtr<T>::get() const noexcept
{
  return internal_ptr({}).get();
}

template <typename T>
typename SharedPtr<T>::element_type& SharedPtr<T>::operator*() const noexcept
{
  return internal_ptr({}).operator*();
}

template <typename T>
typename SharedPtr<T>::element_type* SharedPtr<T>::operator->() const noexcept
{
  return internal_ptr({}).operator->();
}

template <typename T>
typename SharedPtr<T>::ref_count_type SharedPtr<T>::use_count() const noexcept
{
  return internal_ptr({}).use_count();
}

template <typename T>
SharedPtr<T>::operator bool() const noexcept
{
  return internal_ptr({}).operator bool();
}

template <typename T>
typename SharedPtr<T>::internal_ptr_type& SharedPtr<T>::internal_ptr(
  InternalSharedPtrAccessTag) noexcept
{
  return ptr_;
}

template <typename T>
const typename SharedPtr<T>::internal_ptr_type& SharedPtr<T>::internal_ptr(
  InternalSharedPtrAccessTag) const noexcept
{
  return ptr_;
}

// ==========================================================================================

template <typename T, typename U>
bool operator==(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept
{
  return lhs.get() == rhs.get();
}

template <typename T, typename U>
bool operator!=(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept
{
  return lhs.get() != rhs.get();
}

template <typename T, typename U>
bool operator<(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept
{
  return lhs.get() < rhs.get();
}

template <typename T, typename U>
bool operator>(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept
{
  return lhs.get() > rhs.get();
}

template <typename T, typename U>
bool operator<=(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept
{
  return lhs.get() <= rhs.get();
}

template <typename T, typename U>
bool operator>=(const SharedPtr<T>& lhs, const SharedPtr<U>& rhs) noexcept
{
  return lhs.get() >= rhs.get();
}

// ==========================================================================================

template <typename T>
bool operator==(const SharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() == nullptr;
}

template <typename T>
bool operator==(std::nullptr_t, const SharedPtr<T>& rhs) noexcept
{
  return nullptr == rhs.get();
}

template <typename T>
bool operator!=(const SharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() != nullptr;
}

template <typename T>
bool operator!=(std::nullptr_t, const SharedPtr<T>& rhs) noexcept
{
  return nullptr != rhs.get();
}

template <typename T>
bool operator<(const SharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() < nullptr;
}

template <typename T>
bool operator<(std::nullptr_t, const SharedPtr<T>& rhs) noexcept
{
  return nullptr < rhs.get();
}

template <typename T>
bool operator>(const SharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() > nullptr;
}

template <typename T>
bool operator>(std::nullptr_t, const SharedPtr<T>& rhs) noexcept
{
  return nullptr > rhs.get();
}

template <typename T>
bool operator<=(const SharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() <= nullptr;
}

template <typename T>
bool operator<=(std::nullptr_t, const SharedPtr<T>& rhs) noexcept
{
  return nullptr <= rhs.get();
}

template <typename T>
bool operator>=(const SharedPtr<T>& lhs, std::nullptr_t) noexcept
{
  return lhs.get() >= nullptr;
}

template <typename T>
bool operator>=(std::nullptr_t, const SharedPtr<T>& rhs) noexcept
{
  return nullptr >= rhs.get();
}

// ==========================================================================================

template <typename T, typename... Args>
SharedPtr<T> make_shared(Args&&... args)
{
  return SharedPtr<T>{make_internal_shared<T>(std::forward<Args>(args)...)};
}

}  // namespace legate

namespace std {

template <typename T>
std::size_t hash<legate::SharedPtr<T>>::operator()(const legate::SharedPtr<T>& ptr) const noexcept
{
  return hash<typename legate::SharedPtr<T>::element_type*>{}(ptr.get());
}

}  // namespace std
