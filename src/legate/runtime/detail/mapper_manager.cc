/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "legate/runtime/detail/mapper_manager.h"

#include "legate/mapping/detail/base_mapper.h"
#include "legate/runtime/detail/config.h"
#include "legate/runtime/detail/runtime.h"

#include <mappers/logging_wrapper.h>

namespace legate::detail {

MapperManager::MapperManager(Legion::Runtime* legion_runtime)
  : mapper_id_{legion_runtime->generate_library_mapper_ids(
      Runtime::get_runtime()->core_library()->get_library_name().data(), 1)}
{
  auto* const mapper = [&]() -> Legion::Mapping::Mapper* {
    // Legion requires the mapper pointer to be created with new, and takes ownership of it in
    // the call to add_mapper().
    const auto base_mapper = new mapping::detail::BaseMapper{};

    LEGATE_ASSERT(Config::parsed());
    if (Config::log_mapping_decisions) {
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
