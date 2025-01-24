/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
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
