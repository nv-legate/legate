/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

namespace legate::detail {

inline Strategy::Strategy(const Operation* operation) : operation_{operation} {}

inline bool Strategy::parallel() const { return launch_domain_.is_valid(); }

inline const Domain& Strategy::launch_domain() const { return launch_domain_; }

inline void Strategy::set_launch_domain(const Domain& launch_domain)
{
  launch_domain_ = launch_domain;
}

inline void Strategy::insert_store_projection(PrivateKey,
                                              const Variable& partition_symbol,
                                              Legion::ProjectionID projection_id)
{
  projection_ids_.insert({partition_symbol, projection_id});
}

}  // namespace legate::detail
