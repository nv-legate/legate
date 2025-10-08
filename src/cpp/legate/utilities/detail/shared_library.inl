/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/shared_library.h>

namespace legate::detail {

inline ZStringView SharedLibrary::handle_path() const { return handle_path_; }

inline void* SharedLibrary::handle() const { return handle_.get(); }

inline bool SharedLibrary::is_loaded() const { return handle() != nullptr; }

template <typename T>
void SharedLibrary::load_symbol_into(ZStringView symbol_name, T** dest) const
{
  *dest = reinterpret_cast<T*>(load_symbol(symbol_name));
}

}  // namespace legate::detail
