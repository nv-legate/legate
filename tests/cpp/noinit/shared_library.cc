/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/detail/shared_library.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <dlfcn.h>
#include <stdexcept>
#include <utilities/utilities.h>

namespace shared_library_test {

using SharedLibraryUnit = DefaultFixture;

TEST_F(SharedLibraryUnit, MustLoadFailThrows)
{
  const auto make_lib = [] {
    return legate::detail::SharedLibrary{
      "/nonexistent/path/to/libnonexistent.so", RTLD_LAZY | RTLD_LOCAL, /*must_load=*/true};
  };

  ASSERT_THAT(make_lib,
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("Failed to load dynamic shared object")));
}

TEST_F(SharedLibraryUnit, LoadSymbolNotLoadedThrows)
{
  const legate::detail::SharedLibrary lib{
    "/nonexistent/path/to/libnonexistent.so", RTLD_LAZY | RTLD_LOCAL, /*must_load=*/false};

  ASSERT_FALSE(lib.is_loaded());
  ASSERT_THAT([&] { static_cast<void>(lib.load_symbol("any_symbol")); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("failed to load shared library")));
}

TEST_F(SharedLibraryUnit, LoadSymbolUnknownThrows)
{
  const legate::detail::SharedLibrary lib{nullptr};

  ASSERT_TRUE(lib.is_loaded());
  ASSERT_THAT([&] { static_cast<void>(lib.load_symbol("definitely_not_a_real_symbol_xyz_123")); },
              ::testing::ThrowsMessage<std::invalid_argument>(
                ::testing::HasSubstr("Failed to locate the symbol")));
}

TEST_F(SharedLibraryUnit, ExistsNonexistent)
{
  ASSERT_FALSE(legate::detail::SharedLibrary::exists("/nonexistent/path/to/libnonexistent.so"));
}

TEST_F(SharedLibraryUnit, ExistsSystemLib)
{
  ASSERT_TRUE(legate::detail::SharedLibrary::exists("libc.so.6"));
}

}  // namespace shared_library_test
