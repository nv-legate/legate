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
inline void SharedPtr<T>::reference_() noexcept
{
  ptr_.user_reference_({});
}

template <typename T>
inline void SharedPtr<T>::dereference_() noexcept
{
  ptr_.user_dereference_({});
}

template <typename T>
template <typename U>
inline SharedPtr<T>::SharedPtr(copy_tag, const InternalSharedPtr<U>& other) noexcept : ptr_{other}
{
  reference_();
}

template <typename T>
template <typename U>
inline SharedPtr<T>::SharedPtr(move_tag,
                               InternalSharedPtr<U>&& other,
                               bool from_internal_ptr) noexcept
  : ptr_{std::move(other)}
{
  // Only update refcount if we are constructing from a bare InternalSharedPointer, since
  // the previous owning SharedPtr gives ownership entirely
  if (from_internal_ptr) reference_();
}

template <typename T>
template <typename U>
inline void SharedPtr<T>::assign_(copy_tag, const InternalSharedPtr<U>& other) noexcept
{
  SharedPtr tmp{other};

  swap(tmp);
}

template <typename T>
template <typename U>
inline void SharedPtr<T>::assign_(move_tag, InternalSharedPtr<U>&& other) noexcept
{
  SharedPtr tmp{std::move(other)};

  swap(tmp);
}

// ==========================================================================================

template <typename T>
inline SharedPtr<T>::SharedPtr(std::nullptr_t) noexcept
{
}

template <typename T>
template <typename U, typename Deleter, typename Alloc>
inline SharedPtr<T>::SharedPtr(U* ptr, Deleter deleter, Alloc allocator)
  : ptr_{ptr, std::move(deleter), std::move(allocator)}
{
  reference_();
}

template <typename T>
template <typename U>
inline SharedPtr<T>::SharedPtr(U* ptr) : SharedPtr{ptr, std::default_delete<U>{}}
{
}

template <typename T>
inline SharedPtr<T>::SharedPtr(const SharedPtr& other) noexcept : SharedPtr{copy_tag{}, other.ptr_}
{
}

template <typename T>
inline SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr& other) noexcept
{
  assign_(copy_tag{}, other.ptr_);
  return *this;
}

template <typename T>
inline SharedPtr<T>::SharedPtr(SharedPtr&& other) noexcept
  : SharedPtr{move_tag{}, std::move(other.ptr_), false}
{
}

template <typename T>
inline SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr&& other) noexcept
{
  assign_(move_tag{}, std::move(other.ptr_));
  return *this;
}

template <typename T>
template <typename U>
inline SharedPtr<T>::SharedPtr(const SharedPtr<U>& other) noexcept
  : SharedPtr{copy_tag{}, other.ptr_}
{
}

template <typename T>
template <typename U>
inline SharedPtr<T>& SharedPtr<T>::operator=(const SharedPtr<U>& other) noexcept
{
  assign_(copy_tag{}, other.ptr_);
  return *this;
}

template <typename T>
template <typename U>
inline SharedPtr<T>::SharedPtr(SharedPtr<U>&& other) noexcept
  : SharedPtr{move_tag{}, std::move(other.ptr_), false}
{
}

template <typename T>
template <typename U>
inline SharedPtr<T>& SharedPtr<T>::operator=(SharedPtr<U>&& other) noexcept
{
  assign_(move_tag{}, std::move(other.ptr_));
  return *this;
}

template <typename T>
inline SharedPtr<T>::SharedPtr(const InternalSharedPtr<element_type>& other) noexcept
  : SharedPtr{copy_tag{}, other}
{
}

template <typename T>
inline SharedPtr<T>& SharedPtr<T>::operator=(const InternalSharedPtr<element_type>& other) noexcept
{
  assign_(copy_tag{}, other);
  return *this;
}

template <typename T>
inline SharedPtr<T>::SharedPtr(InternalSharedPtr<element_type>&& other) noexcept
  : SharedPtr{move_tag{}, std::move(other), true}
{
}

template <typename T>
inline SharedPtr<T>& SharedPtr<T>::operator=(InternalSharedPtr<element_type>&& other) noexcept
{
  assign_(move_tag{}, std::move(other));
  return *this;
}

template <typename T>
template <typename U>
inline SharedPtr<T>::SharedPtr(const InternalSharedPtr<U>& other) noexcept
  : SharedPtr{copy_tag{}, other}
{
}

template <typename T>
template <typename U>
inline SharedPtr<T>& SharedPtr<T>::operator=(const InternalSharedPtr<U>& other) noexcept
{
  assign_(copy_tag{}, other);
  return *this;
}

template <typename T>
template <typename U>
inline SharedPtr<T>::SharedPtr(InternalSharedPtr<U>&& other) noexcept
  : SharedPtr{move_tag{}, std::move(other), true}
{
}

template <typename T>
template <typename U>
inline SharedPtr<T>& SharedPtr<T>::operator=(InternalSharedPtr<U>&& other) noexcept
{
  assign_(move_tag{}, std::move(other));
  return *this;
}

template <typename T>
template <typename U, typename D>
inline SharedPtr<T>::SharedPtr(std::unique_ptr<U, D>&& ptr) : ptr_{std::move(ptr)}
{
  reference_();
}

template <typename T>
template <typename U, typename D>
inline SharedPtr<T>& SharedPtr<T>::operator=(std::unique_ptr<U, D>&& ptr)
{
  ptr_ = std::move(ptr);
  reference_();
  return *this;
}

template <typename T>
inline SharedPtr<T>::~SharedPtr() noexcept
{
  dereference_();
}

// ==========================================================================================

template <typename T>
inline void SharedPtr<T>::swap(SharedPtr& other) noexcept
{
  ptr_.swap(other.ptr_);
}

// friend function
template <typename T>
inline void swap(SharedPtr<T>& lhs, SharedPtr<T>& rhs) noexcept
{
  lhs.swap(rhs);
}

template <typename T>
inline void SharedPtr<T>::reset() noexcept
{
  SharedPtr<T>{}.swap(*this);
}

template <typename T>
inline void SharedPtr<T>::reset(std::nullptr_t) noexcept
{
  SharedPtr<T>{nullptr}.swap(*this);
}

template <typename T>
template <typename U, typename D, typename A>
inline void SharedPtr<T>::reset(U* ptr, D deleter, A allocator)
{
  // cannot call ptr_.reset() since we may need to bump the user and strong reference counts
  SharedPtr<T>{ptr, std::move(deleter), std::move(allocator)}.swap(*this);
}

// ==========================================================================================

template <typename T>
inline typename SharedPtr<T>::element_type* SharedPtr<T>::get() const noexcept
{
  return ptr_.get();
}

template <typename T>
inline typename SharedPtr<T>::element_type& SharedPtr<T>::operator*() const noexcept
{
  return ptr_.operator*();
}

template <typename T>
inline typename SharedPtr<T>::element_type* SharedPtr<T>::operator->() const noexcept
{
  return ptr_.operator->();
}

template <typename T>
inline typename SharedPtr<T>::ref_count_type SharedPtr<T>::use_count() const noexcept
{
  return ptr_.use_count();
}

template <typename T>
inline SharedPtr<T>::operator bool() const noexcept
{
  return ptr_.operator bool();
}

// ==========================================================================================

template <typename T, typename... Args>
inline SharedPtr<T> make_shared(Args&&... args)
{
  return SharedPtr<T>{make_internal_shared<T>(std::forward<Args>(args)...)};
}

}  // namespace legate
