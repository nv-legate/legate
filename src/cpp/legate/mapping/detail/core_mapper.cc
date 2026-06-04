/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/mapping/detail/core_mapper.h>

#include <legate/mapping/detail/machine.h>
#include <legate/mapping/detail/mapping.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/variant_options.h>
#include <legate/utilities/detail/align.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/env_defaults.h>
#include <legate/utilities/detail/type_traits.h>
#include <legate/utilities/dispatch.h>
#include <legate/utilities/env.h>

#include <cstdlib>
#include <memory>
#include <vector>

namespace legate::mapping::detail {

// This is a custom mapper implementation that only has to map
// start-up tasks associated with Legate, no one else
// should be overriding this mapper so we burry it in here
class CoreMapper final : public Mapper {  // NOLINT(misc-use-internal-linkage)
 public:
  [[nodiscard]] std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override;
  [[nodiscard]] legate::Scalar tunable_value(legate::TunableID tunable_id) override;
  [[nodiscard]] std::optional<std::size_t> allocation_pool_size(
    const legate::mapping::Task& task, legate::mapping::StoreTarget memory_kind) override;
};

std::vector<legate::mapping::StoreMapping> CoreMapper::store_mappings(
  const legate::mapping::Task&, const std::vector<StoreTarget>& /*options*/)
{
  return {};
}

Scalar CoreMapper::tunable_value(TunableID /*tunable_id*/)
{
  // Illegal tunable variable
  LEGATE_ABORT("Tunable values are no longer supported");
  return Scalar{0};
}

namespace {

// The INIT_NCCL task creates two buffers of type Payload (which has two uint64_t-typed members,
// i.e. 16 bytes) with num_ranks elements each. Hence the per-rank size is 16 * 2 = 32 bytes.
constexpr std::size_t NCCL_WARMUP_BUFFER_SIZE_PER_RANK = sizeof(std::uint64_t) * 2 * 2;

constexpr std::size_t EXTRA_SCALAR_ALIGNMENT = 16;

class AlignOfPointType {
 public:
  template <std::int32_t NDIM>
  std::size_t operator()() const
  {
    return alignof(Point<NDIM>);
  }
};

}  // namespace

std::optional<std::size_t> CoreMapper::allocation_pool_size(const legate::mapping::Task&,
                                                            legate::mapping::StoreTarget)
{
  return std::nullopt;
}

std::unique_ptr<Mapper> create_core_mapper() { return std::make_unique<CoreMapper>(); }

}  // namespace legate::mapping::detail
