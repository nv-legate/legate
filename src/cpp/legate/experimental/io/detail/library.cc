/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/experimental/io/detail/library.h>

#include <legate/experimental/io/detail/mapper.h>
#include <legate/runtime/library.h>
#include <legate/runtime/resource.h>
#include <legate/runtime/runtime.h>

#include <memory>
#include <optional>

namespace legate::experimental::io::detail {

legate::Library& core_io_library()
{
  static std::optional<legate::Library> io_lib{};

  if (!io_lib.has_value()) {
    auto* rt = legate::Runtime::get_runtime();

    io_lib.emplace(
      rt->create_library("legate.io", legate::ResourceConfig{}, std::make_unique<Mapper>()));
    rt->register_shutdown_callback([]() noexcept { io_lib.reset(); });
  }
  return *io_lib;
}

}  // namespace legate::experimental::io::detail
