/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/utilities/detail/zstring_view.h>

#include <cstdint>
#include <memory>
#include <string>

namespace legate::detail {

/**
 * @brief RAII wrapper for dynamically loaded shared libraries.
 *
 * Provides functionality to open a shared library, query its load state, and retrieve function
 * or symbol addresses in a type-safe way.
 */
class SharedLibrary {
 public:
  /**
   * @brief Deleted default constructor.
   *
   * A valid library path must always be provided.
   */
  SharedLibrary() = delete;

  /**
   * @brief Constructs and optionally loads a shared library.
   *
   * `lib_path` may be a relative path, absolute path, or just the library name.
   *
   * If a relative path, the library is loaded per the system-specific rules for loading shared
   * objects from relative paths. Usually this means the path is looked up relative to the
   * current working directory.
   *
   * If `lib_path` is just the library name, then the library will be looked up as per the
   * system-specific rules as well. Usually this means that the rpath is searched first, then
   * the global LD caches, and finally the current directory.
   *
   * @param lib_path Path to the shared library file.
   * @param flags Flags controlling library loading (e.g., RTLD_NOW).
   * @param must_load If `true`, the constructor enforces successful loading.
   *
   * @throw std::runtime_error If `must_load` is `true` and the library fails to load.
   */
  explicit SharedLibrary(std::string lib_path, std::int32_t flags, bool must_load);

  /**
   * @brief Constructs and optionally loads a shared library with default flags.
   *
   * @param lib_path Path to the shared library file.
   * @param must_load If `true`, the constructor enforces successful loading.
   *
   * @throw std::runtime_error If `must_load` is `true` and the library fails to load.
   */
  explicit SharedLibrary(std::string lib_path, bool must_load);

  /**
   * @brief Construct a shared library referencing the current shared object.
   *
   * This call generally cannot fail
   */
  explicit SharedLibrary(std::nullptr_t);

  /**
   * If the library did not successfully load, the handle path will be identical to the handle
   * path passed in the constructor.
   *
   * Due to quirks in `dlopen()`, it is not possible to determine the full path to the loaded
   * shared object until you successfully load a symbol from the library. As a result, when a
   * symbol is loaded successfully for the first time (via `load_symbol()` or
   * `load_symbol_into()`), the handle path will be transparently updated to the absolute path
   * of the shared object it was loaded from.
   *
   * @return Path string of the loaded library handle.
   */
  [[nodiscard]] ZStringView handle_path() const;

  /**
   * @return Pointer to the platform-specific library handle.
   */
  [[nodiscard]] void* handle() const;

  /**
   * @return True if the library is loaded, false otherwise.
   */
  [[nodiscard]] bool is_loaded() const;

  /**
   * @brief Loads a symbol by name.
   *
   * @param symbol_name Name of the symbol to resolve.
   *
   * @return Pointer to the resolved symbol.
   *
   * @throw std::runtime_error If the shared library was not successfully loaded before this
   * function was called.
   * @throw std::invalid_argument If the symbol failed to be located in the shared library.
   */
  [[nodiscard]] void* load_symbol(ZStringView symbol_name) const;

  /**
   * @brief Loads a symbol and assigns it to a destination pointer.
   *
   * This is a convenience wrapper for `load_symbol()` to cast the result to the correct
   * type. It shares the same behavior as the former.
   *
   * @param symbol_name Name of the symbol to resolve.
   * @param dest Destination pointer that receives the resolved symbol.
   */
  template <typename T>
  void load_symbol_into(ZStringView symbol_name, T** dest) const;

  /**
   * @brief Checks whether a shared library would be loadable given a particular path.
   *
   * @param lib_path Path to the shared library file.
   *
   * @return True if the shared library exists and is loadable, false otherwise.
   */
  [[nodiscard]] static bool exists(ZStringView lib_path);

 private:
  mutable std::string handle_path_{};
  mutable bool resolved_path_{};
  std::unique_ptr<void, int (*)(void*)> handle_;
};

}  // namespace legate::detail

#include <legate/utilities/detail/shared_library.inl>
