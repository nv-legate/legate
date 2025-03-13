/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/variant_options.h>
#include <legate/utilities/assert.h>

namespace legate {

constexpr VariantOptions& VariantOptions::with_concurrent(bool _concurrent)
{
  concurrent = _concurrent;
  return *this;
}

constexpr VariantOptions& VariantOptions::with_has_allocations(bool _has_allocations)
{
  has_allocations = _has_allocations;
  return *this;
}

constexpr VariantOptions& VariantOptions::with_elide_device_ctx_sync(bool elide_sync)
{
  elide_device_ctx_sync = elide_sync;
  return *this;
}

constexpr VariantOptions& VariantOptions::with_has_side_effect(bool side_effect)
{
  has_side_effect = side_effect;
  return *this;
}

constexpr VariantOptions& VariantOptions::with_may_throw_exception(bool may_throw)
{
  may_throw_exception = may_throw;
  return *this;
}

inline VariantOptions& VariantOptions::with_communicators(
  std::initializer_list<std::string_view> comms) noexcept
{
  return with_communicators({}, std::begin(comms), std::end(comms));
}

template <typename It>
inline VariantOptions& VariantOptions::with_communicators(WithCommunicatorsAccessKey,
                                                          It begin,
                                                          It end) noexcept
{
  if (!communicators.has_value()) {
    communicators.emplace();
  }

  std::size_t i = 0;

  for (; begin != end; ++begin, ++i) {
    LEGATE_CHECK(i < MAX_COMMS);
    (*communicators)[i] = *begin;
  }
  // Clear the rest. Internally an empty communicator is used as the sentinel value.
  for (; i < communicators->size(); ++i) {
    (*communicators)[i] = std::string_view{};
  }
  return with_concurrent(true);
}

constexpr bool VariantOptions::operator==(const VariantOptions& other) const
{
  return concurrent == other.concurrent && has_allocations == other.has_allocations &&
         elide_device_ctx_sync == other.elide_device_ctx_sync &&
         has_side_effect == other.has_side_effect && communicators == other.communicators;
}

constexpr bool VariantOptions::operator!=(const VariantOptions& other) const
{
  return !(*this == other);
}

}  // namespace legate
