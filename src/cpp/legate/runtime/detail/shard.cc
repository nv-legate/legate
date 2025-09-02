/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/shard.h>

#include <legate/runtime/detail/library.h>
#include <legate/runtime/detail/projection.h>
#include <legate/utilities/detail/core_ids.h>
#include <legate/utilities/detail/linearize.h>
#include <legate/utilities/detail/type_traits.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace legate::detail {

namespace {

std::unordered_map<Legion::ProjectionID, Legion::ShardID> functor_id_table{};
std::mutex functor_table_lock{};

}  // namespace

class ToplevelTaskShardingFunctor final : public Legion::ShardingFunctor {
 public:
  [[nodiscard]] Legion::ShardID shard(const DomainPoint& p,
                                      const Domain& launch_space,
                                      const std::size_t total_shards) override
  {
    // Just tile this space in 1D
    const Point<1> point    = p;
    const Rect<1> space     = launch_space;
    const std::size_t size  = (space.hi[0] - space.lo[0]) + 1;
    const std::size_t chunk = (size + total_shards - 1) / total_shards;
    return (point[0] - space.lo[0]) / chunk;
  }
};

class LinearizingShardingFunctor final : public Legion::ShardingFunctor {
 public:
  [[nodiscard]] Legion::ShardID shard(const DomainPoint& p,
                                      const Domain& launch_space,
                                      const std::size_t total_shards) override
  {
    const std::size_t size  = launch_space.get_volume();
    const std::size_t chunk = (size + total_shards - 1) / total_shards;
    return linearize(launch_space.lo(), launch_space.hi(), p) / chunk;
  }

  [[nodiscard]] bool is_invertible() const override { return true; }

  void invert(Legion::ShardID shard,
              const Domain& shard_domain,
              const Domain& full_domain,
              const std::size_t total_shards,
              std::vector<DomainPoint>& points) override
  {
    LEGATE_CHECK(shard_domain == full_domain);
    const std::size_t size  = shard_domain.get_volume();
    const std::size_t chunk = (size + total_shards - 1) / total_shards;
    std::size_t idx         = shard * chunk;
    const std::size_t lim   = std::min((shard + 1) * chunk, size);
    if (idx >= lim) {
      return;
    }

    auto point = delinearize(shard_domain.lo(), shard_domain.hi(), idx);

    points.reserve(points.size() + lim);
    for (; idx < lim; ++idx) {
      points.push_back(point);
      for (int dim = shard_domain.dim - 1; dim >= 0; --dim) {
        LEGATE_CHECK(shard_domain.dim <= Domain::MAX_RECT_DIM);
        if (point[dim] < shard_domain.hi()[dim]) {
          point[dim]++;
          break;
        }
        point[dim] = shard_domain.lo()[dim];
      }
    }
  }
};

void register_legate_core_sharding_functors(const detail::Library& core_library)
{
  auto runtime = Legion::Runtime::get_runtime();

  runtime->register_sharding_functor(
    core_library.get_sharding_id(to_underlying(CoreShardID::TOPLEVEL_TASK)),
    new ToplevelTaskShardingFunctor{},
    true /*silence warnings*/);

  auto sharding_id = core_library.get_sharding_id(to_underlying(CoreShardID::LINEARIZE));
  runtime->register_sharding_functor(
    sharding_id, new LinearizingShardingFunctor{}, true /*silence warnings*/);
  // Use linearizing functor for identity projections
  functor_id_table[0] = sharding_id;
}

class LegateShardingFunctor final : public Legion::ShardingFunctor {
 public:
  LegateShardingFunctor(ProjectionFunction* proj_fn, const mapping::ProcessorRange& range)
    : proj_fn_{proj_fn}, range_{range}
  {
  }

  [[nodiscard]] Legion::ShardID shard(const DomainPoint& p,
                                      const Domain& launch_space,
                                      std::size_t total_shards) override
  {
    auto lo             = proj_fn_->project_point(launch_space.lo());
    auto hi             = proj_fn_->project_point(launch_space.hi());
    auto point          = proj_fn_->project_point(p);
    auto task_count     = linearize(lo, hi, hi) + 1;
    auto global_proc_id = ((linearize(lo, hi, point) * range_.count()) / task_count) + range_.low;
    auto shard_id       = global_proc_id / range_.per_node_count;
    LEGATE_ASSERT(shard_id < total_shards);
    return static_cast<Legion::ShardID>(shard_id);
  }

 private:
  ProjectionFunction* proj_fn_{};
  mapping::ProcessorRange range_{};
};

Legion::ShardingID find_sharding_functor_by_projection_functor(Legion::ProjectionID proj_id)
{
  const std::scoped_lock<std::mutex> lock{functor_table_lock};
  const auto it = functor_id_table.find(proj_id);

  LEGATE_CHECK(it != functor_id_table.end());
  return it->second;
}

class ShardingCallbackArgs {
 public:
  Legion::ShardID shard_id{};
  Legion::ProjectionID proj_id{};
  mapping::ProcessorRange range{};
};

namespace {

void sharding_functor_registration_callback(const Legion::RegistrationCallbackArgs& args)
{
  auto p_args           = static_cast<ShardingCallbackArgs*>(args.buffer.get_ptr());
  auto runtime          = Legion::Runtime::get_runtime();
  auto sharding_functor = std::make_unique<LegateShardingFunctor>(
    find_projection_function(p_args->proj_id), p_args->range);
  runtime->register_sharding_functor(
    p_args->shard_id, sharding_functor.release(), true /*silence warnings*/);
}

}  // namespace

void create_sharding_functor_using_projection(Legion::ShardID shard_id,
                                              Legion::ProjectionID proj_id,
                                              const mapping::ProcessorRange& range)
{
  legate::detail::ShardingCallbackArgs args{shard_id, proj_id, range};
  {
    const std::scoped_lock<std::mutex> lock{legate::detail::functor_table_lock};
    legate::detail::functor_id_table[proj_id] = shard_id;
  }
  const Legion::UntypedBuffer buffer{&args, sizeof(args)};
  Legion::Runtime::perform_registration_callback(
    legate::detail::sharding_functor_registration_callback,
    buffer,
    false /*global*/,
    false /*dedup*/);
}

}  // namespace legate::detail
