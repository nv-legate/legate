/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/mapper_manager.h>

#include <legate/mapping/detail/base_mapper.h>
#include <legate/runtime/detail/config.h>
#include <legate/runtime/detail/runtime.h>

#include <mappers/logging_wrapper.h>

namespace legate::detail {

MapperManager::MapperManager(Legion::Runtime* legion_runtime)
  : mapper_id_{legion_runtime->generate_library_mapper_ids(
      // get_library_name() returns a ZStringView which is guaranteed to be NULL-terminated
      Runtime::get_runtime()
        ->core_library()
        ->get_library_name()
        .data(),  // NOLINT(bugprone-suspicious-stringview-data-usage)
      1)}
{
  auto* const mapper = [&]() -> Legion::Mapping::Mapper* {
    // Legion requires the mapper pointer to be created with new, and takes ownership of it in
    // the call to add_mapper().
    const auto base_mapper = new mapping::detail::BaseMapper{};

    LEGATE_ASSERT(Config::get_config().parsed());
    if (Config::get_config().log_mapping_decisions()) {
      try {
        return new Legion::Mapping::LoggingWrapper{base_mapper, &base_mapper->logger()};
      } catch (...) {
        delete base_mapper;
        throw;
      }
    }
    return base_mapper;
  }();

  // No try-catch around this. If it fails, Legion will just abort the program.
  legion_runtime->add_mapper(mapper_id(), mapper);
}

// ==========================================================================================

MapperManager::MapperManager() : MapperManager{Legion::Runtime::get_runtime()} {}

}  // namespace legate::detail
