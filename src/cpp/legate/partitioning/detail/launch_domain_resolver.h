/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/partitioning/detail/restriction.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <optional>

namespace legate::detail {

class LaunchDomainResolver {
 public:
  void record_launch_domain(const Domain& launch_domain);
  void record_unbound_store(std::uint32_t unbound_dim);
  void set_must_be_sequential(bool must_be_sequential);

  [[nodiscard]] Domain resolve_launch_domain() const;

 private:
  bool must_be_sequential_{};
  bool must_be_1d_{};
  std::optional<std::uint32_t> unbound_dim_{};
  std::optional<Domain> launch_domain_{};
  std::optional<std::size_t> launch_volume_{};
};

}  // namespace legate::detail
