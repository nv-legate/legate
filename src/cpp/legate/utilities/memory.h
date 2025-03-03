/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <memory>

namespace legate {

/// @brief deleter for using unique_ptr with incomplete types
/// @tparam T the type to delete
/// @code{.cpp}
///  // in header file:
///  struct Foo;
///  extern template class legate::DefaultDelete<Foo>; // Suppress instantiation
///  std::unique_ptr<Foo, DefaultDelete<Foo>> foo;     // OK
///
///  // in source file:
///  struct Foo { int x; };
///  template class legate::DefaultDelete<Foo>;        // Explicit instantiation
/// @endcode
template <typename T>
class DefaultDelete {
 public:
  void operator()(T*) const noexcept;
};

}  // namespace legate

#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/shared_ptr.h>

#include <legate/utilities/memory.inl>
