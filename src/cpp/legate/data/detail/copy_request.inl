/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/copy_request.h>

namespace legate::detail {

inline CopyRequest::CopyRequest(std::uint64_t storage_id,
                                InternalSharedPtr<InlineStorage> source,
                                InternalSharedPtr<InlineStorage> target,
                                InternalSharedPtr<Partition> source_partition,
                                InternalSharedPtr<Partition> target_partition,
                                const Rect& source_domain,
                                const Rect& target_domain,
                                std::uint32_t field_size)
  : storage_id_{storage_id},
    source_{std::move(source)},
    target_{std::move(target)},
    source_partition_{std::move(source_partition)},
    target_partition_{std::move(target_partition)},
    source_domain_{source_domain},
    target_domain_{target_domain},
    field_size_{field_size}
{
}

}  // namespace legate::detail
