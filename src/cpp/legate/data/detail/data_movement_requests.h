/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/copy_request.h>
#include <legate/data/detail/reduction_request.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>

namespace legate::detail {

class DataMovementRequests {
 public:
  void register_copy_in(CopyRequest&& copy_in);
  void register_reduction(ReductionRequest&& reduction);

  void require_data_sync();

  void reserve_copy_ins(std::size_t size);
  void reserve_reduction_requests(std::size_t size);

  void perform_data_sync() const;
  void perform_copy_ins(VariantCode variant_code) const;
  void perform_reductions(VariantCode variant_code) const;

  bool empty() const;

 private:
  bool data_sync_required_{};
  std::vector<CopyRequest> copy_ins_{};
  std::vector<ReductionRequest> reductions_{};
};

}  // namespace legate::detail

#include <legate/data/detail/data_movement_requests.inl>
