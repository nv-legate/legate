/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/reduction_request.h>

namespace legate::detail {

inline ReductionRequest::ReductionRequest(InternalSharedPtr<InlineStorage> source,
                                          InternalSharedPtr<InlineStorage> target,
                                          InternalSharedPtr<Type> type,
                                          ReductionOpKind op_kind)
  : source_{std::move(source)},
    target_{std::move(target)},
    type_{std::move(type)},
    op_kind_{op_kind}
{
}

}  // namespace legate::detail
