/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/inline_storage.h>
#include <legate/type/detail/types.h>
#include <legate/type/types.h>
#include <legate/utilities/internal_shared_ptr.h>
#include <legate/utilities/typedefs.h>

namespace legate::detail {

// TODO(wonchanl): Only support 1d reductions on unpartitioned stores, should be extended to other
// cases.
class ReductionRequest {
 public:
  ReductionRequest(InternalSharedPtr<InlineStorage> source,
                   InternalSharedPtr<InlineStorage> target,
                   InternalSharedPtr<Type> type,
                   ReductionOpKind op_kind);
  void perform(VariantCode variant_code) const;

  ReductionRequest(const ReductionRequest&)            = default;
  ReductionRequest& operator=(const ReductionRequest&) = default;

  ReductionRequest(ReductionRequest&&) noexcept            = default;
  ReductionRequest& operator=(ReductionRequest&&) noexcept = default;

 private:
  InternalSharedPtr<InlineStorage> source_{};
  InternalSharedPtr<InlineStorage> target_{};
  InternalSharedPtr<Type> type_{};
  ReductionOpKind op_kind_{};
};

}  // namespace legate::detail

#include <legate/data/detail/copy_request.inl>
