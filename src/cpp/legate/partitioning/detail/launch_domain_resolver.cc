/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/partitioning/detail/launch_domain_resolver.h>

#include <legate/utilities/assert.h>
#include <legate/utilities/typedefs.h>

#include <cstdint>
#include <optional>

namespace legate::detail {

void LaunchDomainResolver::record_launch_domain(const Rect& launch_domain)
{
  const auto new_volume = launch_domain.volume();

  if (!launch_volume_.has_value()) {
    LEGATE_ASSERT(!launch_domain_.has_value());
    launch_volume_.emplace(new_volume);
    launch_domain_.emplace(launch_domain);
    return;
  }

  if (*launch_volume_ != new_volume) {
    must_be_sequential_ = true;
    return;
  }

  LEGATE_ASSERT(launch_domain_.has_value());
  if (*launch_domain_ != launch_domain) {
    must_be_1d_ = true;
  }
}

void LaunchDomainResolver::record_unbound_store(std::uint32_t unbound_dim)
{
  if (unbound_dim_.has_value() && *unbound_dim_ != unbound_dim) {
    set_must_be_sequential(true);
  } else {
    unbound_dim_ = unbound_dim;
  }
}

void LaunchDomainResolver::set_must_be_sequential(bool must_be_sequential)
{
  must_be_sequential_ = must_be_sequential;
}

Rect LaunchDomainResolver::resolve_launch_domain() const
{
  if (must_be_sequential_ || !launch_domain_.has_value()) {
    return {};
  }
  if (must_be_1d_) {
    if (unbound_dim_.value_or(0) > 1) {
      return {};
    }
    LEGATE_ASSERT(launch_volume_ >= 1);
    return {
      0,
      static_cast<coord_t>(*launch_volume_ - 1)  // NOLINT(bugprone-unchecked-optional-access)
    };
  }

  LEGATE_ASSERT(launch_domain_.has_value());

  const auto& launch_domain = *launch_domain_;

  if (unbound_dim_.has_value() && launch_domain.dim() != *unbound_dim_) {
    return {};
  }
  return launch_domain;
}

}  // namespace legate::detail
