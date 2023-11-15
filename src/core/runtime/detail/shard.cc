/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "core/runtime/detail/shard.h"

#include "core/runtime/detail/library.h"
#include "core/runtime/detail/projection.h"
#include "core/utilities/linearize.h"

#include <algorithm>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace legate::detail {

namespace {

std::unordered_map<Legion::ProjectionID, Legion::ShardID> functor_id_table{};
std::mutex functor_table_lock{};

}  // namespace

class ToplevelTaskShardingFunctor : public Legion::ShardingFunctor {
 public:
  [[nodiscard]] Legion::ShardID shard(const DomainPoint& p,
                                      const Domain& launch_space,
                                      const size_t total_shards) override
  {
    // Just tile this space in 1D
    const Point<1> point = p;
    const Rect<1> space  = launch_space;
    const size_t size    = (space.hi[0] - space.lo[0]) + 1;
    const size_t chunk   = (size + total_shards - 1) / total_shards;
    return (point[0] - space.lo[0]) / chunk;
  }
};

class LinearizingShardingFunctor : public Legion::ShardingFunctor {
 public:
  [[nodiscard]] Legion::ShardID shard(const DomainPoint& p,
                                      const Domain& launch_space,
                                      const size_t total_shards) override
  {
    const size_t size  = launch_space.get_volume();
    const size_t chunk = (size + total_shards - 1) / total_shards;
    return linearize(launch_space.lo(), launch_space.hi(), p) / chunk;
  }

  [[nodiscard]] bool is_invertible() const override { return true; }

  void invert(Legion::ShardID shard,
              const Domain& shard_domain,
              const Domain& full_domain,
              const size_t total_shards,
              std::vector<DomainPoint>& points) override
  {
    assert(shard_domain == full_domain);
    const size_t size  = shard_domain.get_volume();
    const size_t chunk = (size + total_shards - 1) / total_shards;
    size_t idx         = shard * chunk;
    const size_t lim   = std::min((shard + 1) * chunk, size);
    if (idx >= lim) {
      return;
    }

    auto point = delinearize(shard_domain.lo(), shard_domain.hi(), idx);

    points.reserve(points.size() + lim);
    for (; idx < lim; ++idx) {
      points.push_back(point);
      for (int dim = shard_domain.dim - 1; dim >= 0; --dim) {
        if (point[dim] < shard_domain.hi()[dim]) {
          point[dim]++;
          break;
        }
        point[dim] = shard_domain.lo()[dim];
      }
    }
  }
};

void register_legate_core_sharding_functors(Legion::Runtime* runtime,
                                            const detail::Library* core_library)
{
  runtime->register_sharding_functor(
    core_library->get_sharding_id(LEGATE_CORE_TOPLEVEL_TASK_SHARD_ID),
    new ToplevelTaskShardingFunctor{},
    true /*silence warnings*/);

  auto sharding_id = core_library->get_sharding_id(LEGATE_CORE_LINEARIZE_SHARD_ID);
  runtime->register_sharding_functor(
    sharding_id, new LinearizingShardingFunctor{}, true /*silence warnings*/);
  // Use linearizing functor for identity projections
  functor_id_table[0] = sharding_id;
  // and for the delinearizing projection
  functor_id_table[core_library->get_projection_id(LEGATE_CORE_DELINEARIZE_PROJ_ID)] = sharding_id;
}

class LegateShardingFunctor : public Legion::ShardingFunctor {
 public:
  LegateShardingFunctor(LegateProjectionFunctor* proj_functor,
                        uint32_t start_proc_id,
                        uint32_t end_proc_id,
                        uint32_t per_node_count)
    : proj_functor_{proj_functor},
      start_proc_id_{start_proc_id},
      end_proc_id_{end_proc_id},
      per_node_count_{per_node_count}
  {
  }

  [[nodiscard]] Legion::ShardID shard(const DomainPoint& p,
                                      const Domain& launch_space,
                                      size_t total_shards) override
  {
    auto lo             = proj_functor_->project_point(launch_space.lo(), launch_space);
    auto hi             = proj_functor_->project_point(launch_space.hi(), launch_space);
    auto point          = proj_functor_->project_point(p, launch_space);
    auto task_count     = linearize(lo, hi, hi) + 1;
    auto proc_count     = end_proc_id_ - start_proc_id_;
    auto global_proc_id = (linearize(lo, hi, point) * proc_count) / task_count + start_proc_id_;
    auto shard_id       = global_proc_id / per_node_count_;
    if (LegateDefined(LEGATE_USE_DEBUG)) {
      assert(shard_id < total_shards);
    }
    return static_cast<Legion::ShardID>(shard_id);
  }

 private:
  LegateProjectionFunctor* proj_functor_{};
  uint32_t start_proc_id_{};
  uint32_t end_proc_id_{};
  uint32_t per_node_count_{};
};

Legion::ShardingID find_sharding_functor_by_projection_functor(Legion::ProjectionID proj_id)
{
  const std::lock_guard<std::mutex> lock{functor_table_lock};
  assert(functor_id_table.find(proj_id) != functor_id_table.end());
  return functor_id_table[proj_id];
}

struct ShardingCallbackArgs {
  Legion::ShardID shard_id{};
  Legion::ProjectionID proj_id{};
  uint32_t start_proc_id{};
  uint32_t end_proc_id{};
  uint32_t per_node_count{};
};

namespace {

void sharding_functor_registration_callback(const Legion::RegistrationCallbackArgs& args)
{
  auto p_args           = static_cast<ShardingCallbackArgs*>(args.buffer.get_ptr());
  auto runtime          = Legion::Runtime::get_runtime();
  auto sharding_functor = new LegateShardingFunctor{find_legate_projection_functor(p_args->proj_id),
                                                    p_args->start_proc_id,
                                                    p_args->end_proc_id,
                                                    p_args->per_node_count};
  runtime->register_sharding_functor(p_args->shard_id, sharding_functor, true /*silence warnings*/);
}

}  // namespace

}  // namespace legate::detail

extern "C" {

void legate_create_sharding_functor_using_projection(Legion::ShardID shard_id,
                                                     Legion::ProjectionID proj_id,
                                                     uint32_t start_proc_id,
                                                     uint32_t end_proc_id,
                                                     uint32_t per_node_count)
{
  legate::detail::ShardingCallbackArgs args{
    shard_id, proj_id, start_proc_id, end_proc_id, per_node_count};
  {
    const std::lock_guard<std::mutex> lock{legate::detail::functor_table_lock};
    legate::detail::functor_id_table[proj_id] = shard_id;
  }
  const Legion::UntypedBuffer buffer{&args, sizeof(args)};
  Legion::Runtime::perform_registration_callback(
    legate::detail::sharding_functor_registration_callback,
    buffer,
    false /*global*/,
    false /*dedup*/);
}
}
