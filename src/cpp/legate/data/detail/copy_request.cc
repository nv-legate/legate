/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/copy_request.h>

#include <legate/comm/coll.h>
#include <legate/comm/coll_comm.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/data/detail/copy_util.h>
#include <legate/data/detail/inline_storage.h>
#include <legate/generated/fatbin/copy_request_fatbin.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/enumerate.h>
#include <legate/utilities/detail/geometry.h>
#include <legate/utilities/detail/small_vector.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/internal_shared_ptr.h>

#include <nccl.h>
#include <stdexcept>

namespace legate::detail {

Logger& log_debug()
{
  static Logger log{"legate.debug"};

  return log;
}

void CopyRequest::perform(VariantCode variant_code) const
{
  LEGATE_CHECK(source_partition_->has_color_shape());

  if (!target_partition_->has_color_shape()) {
    LEGATE_CHECK(variant_code == VariantCode::CPU);

    cpu_all_gather_();
  } else {
    shuffle_(variant_code);
  }
}

namespace {

#define LEGATE_CHECK_NCCL(...)               \
  do {                                       \
    const ncclResult_t result = __VA_ARGS__; \
    check_nccl(result, __FILE__, __LINE__);  \
  } while (false)

void check_nccl(ncclResult_t error, const char* file, int line)
{
  if (error != ncclSuccess) {
    static_cast<void>(fprintf(stderr,
                              "Internal NCCL failure with error %d (%s) in file %s at line %d\n",
                              error,
                              ncclGetErrorString(error),
                              file,
                              line));
    std::exit(error);
  }
}

template <typename VAL>
void cpu_reorder_elements_(InlineStorage& target, const InlineStorage& source, const Tiling* tiling)
{
  auto* target_ptr       = static_cast<VAL*>(target.data());
  const auto* source_ptr = static_cast<const VAL*>(source.data());

  const auto target_strides = calculate_strides(to_shape(target.domain()));

  const auto& tile_shape  = tiling->tile_shape_dp();
  const auto tile_strides = calculate_strides(tile_shape);
  const auto tile_volume  = calculate_volume(tile_shape);

  const auto& color_strides = calculate_strides(tiling->color_shape_dp());

  for (Domain::DomainPointIterator it{target.domain()}; it; ++it) {
    const auto target_idx = inner_product(it.p, target_strides);
    const auto source_idx = inner_product(it.p % tile_shape, tile_strides) +
                            inner_product(it.p / tile_shape, color_strides) * tile_volume;

    target_ptr[target_idx] = source_ptr[source_idx];
  }
}

template <typename VAL>
void cpu_copy_to_send_buffer(InlineStorage& target,
                             const InlineStorage& source,
                             const Domain& source_domain,
                             const DomainPoint& source_strides,
                             const SmallVector<Domain>& target_subdomains,
                             const SmallVector<DomainPoint>& all_target_strides,
                             const SmallVector<int>& counts,
                             const SmallVector<int>& offsets)
{
  const auto node_count = Runtime::get_runtime().node_count();
  const auto* source_ptr =
    static_cast<const VAL*>(source.data()) - inner_product(source_domain.lo(), source_strides);

  for (std::uint32_t idx = 0; idx < node_count; ++idx) {
    if (counts[idx] == 0) {
      continue;
    }

    const auto& target_subdomain = target_subdomains[idx];
    const auto& target_strides   = all_target_strides[idx];
    auto* target_ptr             = static_cast<VAL*>(target.data()) + offsets[idx] -
                                   inner_product(target_subdomain.lo(), target_strides);

    for (Domain::DomainPointIterator it{target_subdomain, false}; it; ++it) {
      target_ptr[inner_product(it.p, target_strides)] =
        source_ptr[inner_product(it.p, source_strides)];

      log_debug().print() << "source_domain: " << source_domain
                          << ", target_subdomain: " << target_subdomain
                          << ", target_idx: " << inner_product(it.p, target_strides)
                          << ", source_idx: " << inner_product(it.p, source_strides);
    }
  }
}

void cpu_copy_to_send_buffer(InlineStorage& target,
                             const InlineStorage& source,
                             const Domain& source_domain,
                             const DomainPoint& source_strides,
                             const SmallVector<Domain>& target_subdomains,
                             const SmallVector<DomainPoint>& all_target_strides,
                             const SmallVector<int>& counts,
                             const SmallVector<int>& offsets,
                             std::uint32_t field_size)
{
  switch (field_size) {
    case 1: {
      cpu_copy_to_send_buffer<std::uint8_t>(target,
                                            source,
                                            source_domain,
                                            source_strides,
                                            target_subdomains,
                                            all_target_strides,
                                            counts,
                                            offsets);
      break;
    }
    case 2: {
      cpu_copy_to_send_buffer<std::uint16_t>(target,
                                             source,
                                             source_domain,
                                             source_strides,
                                             target_subdomains,
                                             all_target_strides,
                                             counts,
                                             offsets);
      break;
    }
    case 4: {
      cpu_copy_to_send_buffer<std::uint32_t>(target,
                                             source,
                                             source_domain,
                                             source_strides,
                                             target_subdomains,
                                             all_target_strides,
                                             counts,
                                             offsets);
      break;
    }
    case 8: {
      cpu_copy_to_send_buffer<std::uint64_t>(target,
                                             source,
                                             source_domain,
                                             source_strides,
                                             target_subdomains,
                                             all_target_strides,
                                             counts,
                                             offsets);
      break;
    }
    default: {  // legate-lint: no-switch-default
      throw TracedException<std::runtime_error>{"Unsupported field size"};
    }
  }
}

template <typename VAL>
void cpu_copy_from_recv_buffer(InlineStorage& target,
                               const Domain& target_domain,
                               const DomainPoint& target_strides,
                               const InlineStorage& source,
                               const SmallVector<Domain>& source_subdomains,
                               const SmallVector<DomainPoint>& all_source_strides,
                               const SmallVector<int>& counts,
                               const SmallVector<int>& offsets)
{
  const auto node_count = Runtime::get_runtime().node_count();

  auto* target_ptr =
    static_cast<VAL*>(target.data()) - inner_product(target_domain.lo(), target_strides);

  for (std::uint32_t idx = 0; idx < node_count; ++idx) {
    if (counts[idx] == 0) {
      continue;
    }

    const auto& source_subdomain = source_subdomains[idx];
    const auto& source_strides   = all_source_strides[idx];
    const auto* source_ptr       = static_cast<const VAL*>(source.data()) + offsets[idx] -
                                   inner_product(source_subdomain.lo(), source_strides);

    for (Domain::DomainPointIterator it{source_subdomain}; it; ++it) {
      target_ptr[inner_product(it.p, target_strides)] =
        source_ptr[inner_product(it.p, source_strides)];
    }
  }
}

void cpu_copy_from_recv_buffer(InlineStorage& target,
                               const Domain& target_domain,
                               const DomainPoint& target_strides,
                               const InlineStorage& source,
                               const SmallVector<Domain>& source_subdomains,
                               const SmallVector<DomainPoint>& all_source_strides,
                               const SmallVector<int>& counts,
                               const SmallVector<int>& offsets,
                               std::uint32_t field_size)
{
  switch (field_size) {
    case 1: {
      cpu_copy_from_recv_buffer<std::uint8_t>(target,
                                              target_domain,
                                              target_strides,
                                              source,
                                              source_subdomains,
                                              all_source_strides,
                                              counts,
                                              offsets);
      break;
    }
    case 2: {
      cpu_copy_from_recv_buffer<std::uint16_t>(target,
                                               target_domain,
                                               target_strides,
                                               source,
                                               source_subdomains,
                                               all_source_strides,
                                               counts,
                                               offsets);
      break;
    }
    case 4: {
      cpu_copy_from_recv_buffer<std::uint32_t>(target,
                                               target_domain,
                                               target_strides,
                                               source,
                                               source_subdomains,
                                               all_source_strides,
                                               counts,
                                               offsets);
      break;
    }
    case 8: {
      cpu_copy_from_recv_buffer<std::uint64_t>(target,
                                               target_domain,
                                               target_strides,
                                               source,
                                               source_subdomains,
                                               all_source_strides,
                                               counts,
                                               offsets);
      break;
    }
    default: {  // legate-lint: no-switch-default
      throw TracedException<std::runtime_error>{"Unsupported field size"};
    }
  }
}

void gpu_copy_to_send_buffer(SmallVector<InternalSharedPtr<InlineStorage>>& targets,
                             const InlineStorage& source,
                             const Domain& source_domain,
                             DomainPoint& source_strides,
                             const SmallVector<Domain>& target_subdomains,
                             SmallVector<DomainPoint>& all_target_strides,
                             const SmallVector<int>& counts,
                             std::uint32_t,
                             CUstream stream)
{
  using VAL = std::uint32_t;

  const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

  auto&& api         = cuda::detail::get_cuda_driver_api();
  auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();
  CUkernel kern =
    mod_manager.load_kernel_from_fatbin(copy_request_fatbin, "legate_copy_subdomain_32");

  const auto node_count = Runtime::get_runtime().node_count();
  const auto* source_ptr =
    static_cast<const VAL*>(source.data()) - inner_product(source_domain.lo(), source_strides);

  for (std::uint32_t idx = 0; idx < node_count; ++idx) {
    if (counts[idx] == 0) {
      continue;
    }

    const auto& target_subdomain = target_subdomains[idx];
    auto& target_strides         = all_target_strides[idx];
    auto target_lo               = target_subdomain.lo();
    auto* target_ptr =
      static_cast<VAL*>(targets[idx]->data()) - inner_product(target_lo, target_strides);

    const auto num_blocks = (static_cast<std::size_t>(counts[idx]) + LEGATE_THREADS_PER_BLOCK - 1) /
                            LEGATE_THREADS_PER_BLOCK;
    auto count            = counts[idx];
    auto target_extents   = calculate_extents(target_subdomain);

    api->launch_kernel(kern,
                       {num_blocks},
                       {LEGATE_THREADS_PER_BLOCK},
                       /*shared_mem_bytes=*/0,
                       stream,
                       static_cast<void*>(target_ptr),
                       static_cast<const void*>(source_ptr),
                       count,
                       target_extents,
                       target_lo,
                       target_strides,
                       source_strides);

    api->stream_synchronize(stream);

    auto host_target = std::malloc(target_subdomain.get_volume() * sizeof(VAL));
    api->mem_cpy_async(
      host_target, targets[idx]->data(), target_subdomain.get_volume() * sizeof(VAL), stream);
  }
}

void gpu_copy_from_recv_buffer(InlineStorage& target,
                               const Domain& target_domain,
                               DomainPoint& target_strides,
                               const SmallVector<InternalSharedPtr<InlineStorage>>& sources,
                               const SmallVector<Domain>& source_subdomains,
                               SmallVector<DomainPoint>& all_source_strides,
                               const SmallVector<int>& counts,
                               std::uint32_t,
                               CUstream stream)
{
  using VAL = std::uint32_t;

  const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

  auto&& api         = cuda::detail::get_cuda_driver_api();
  auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();
  CUkernel kern =
    mod_manager.load_kernel_from_fatbin(copy_request_fatbin, "legate_copy_subdomain_32");

  const auto node_count = Runtime::get_runtime().node_count();

  auto* target_ptr =
    static_cast<VAL*>(target.data()) - inner_product(target_domain.lo(), target_strides);

  for (std::uint32_t idx = 0; idx < node_count; ++idx) {
    if (counts[idx] == 0) {
      continue;
    }

    const auto& source_subdomain = source_subdomains[idx];
    auto& source_strides         = all_source_strides[idx];
    auto source_lo               = source_subdomain.lo();
    const auto* source_ptr =
      static_cast<const VAL*>(sources[idx]->data()) - inner_product(source_lo, source_strides);

    const auto num_blocks = (static_cast<std::size_t>(counts[idx]) + LEGATE_THREADS_PER_BLOCK - 1) /
                            LEGATE_THREADS_PER_BLOCK;
    auto count            = counts[idx];
    auto source_extents   = calculate_extents(source_subdomain);

    api->launch_kernel(kern,
                       {num_blocks},
                       {LEGATE_THREADS_PER_BLOCK},
                       /*shared_mem_bytes=*/0,
                       stream,
                       static_cast<void*>(target_ptr),
                       static_cast<const void*>(source_ptr),
                       count,
                       source_extents,
                       source_lo,
                       target_strides,
                       source_strides);
  }
}

}  // namespace

void CopyRequest::cpu_all_gather_() const
{
  // TODO(wonchanl): The code below doesn't handle the case where there are smaller tiles

  // If the source partition is a "batch" partition, the source can be directly all-gathered into
  // the target in the correct order
  //
  // TODO(wonchanl): need to check if the device mesh has C order as well
  if (source_partition_->color_shape().back() == 1) {
    collAllgather(source_->data(),
                  target_->data(),
                  source_->domain().get_volume(),
                  to_coll_data_type_(field_size_),
                  Runtime::get_runtime().get_cpu_comm());
  }

  // Otherwise, the all-gather result needs to be re-ordered into the target

  InlineStorage tmp{target_->domain(), field_size_, target_->target()};

  collAllgather(source_->data(),
                tmp.data(),
                source_->domain().get_volume(),
                to_coll_data_type_(field_size_),
                Runtime::get_runtime().get_cpu_comm());

  // If we're here, the target partition is NoPartition, so the source partition cannot be
  // NoPartition as well.
  const auto* tiling = static_cast<Tiling*>(source_partition_.get());

  switch (field_size_) {
    // FIXME(wonchanl): source_->domain() may not be the tile shape if source_ isn't a tight
    // allocation
    case 1: {
      cpu_reorder_elements_<std::uint8_t>(*target_, tmp, tiling);
      break;
    }
    case 2: {
      cpu_reorder_elements_<std::uint16_t>(*target_, tmp, tiling);
      break;
    }
    case 4: {
      cpu_reorder_elements_<std::uint32_t>(*target_, tmp, tiling);
      break;
    }
    case 8: {
      cpu_reorder_elements_<std::uint64_t>(*target_, tmp, tiling);
      break;
    }
    default: {  // legate-lint: no-switch-default
      throw TracedException<std::runtime_error>{"Unsupported field size"};
    }
  }
}

void CopyRequest::cpu_shuffle_() const
{
  // If the source_partition_ was a NoPartition, the subsumption check would have passed.
  LEGATE_ASSERT(dynamic_cast<Tiling*>(source_partition_.get()));

  const auto* source_tiling = static_cast<Tiling*>(source_partition_.get());
  const auto* target_tiling = static_cast<Tiling*>(target_partition_.get());

  const auto node_id = Runtime::get_runtime().node_id();

  const auto target_subdomain =
    target_tiling->shard(sub_domain_, target_tiling->compute_color(node_id));
  const auto source_subdomain =
    source_tiling->shard(sub_domain_, source_tiling->compute_color(node_id));

  const auto& source_tile_shape = source_tiling->tile_shape_dp();
  const auto source_colors =
    Domain{target_subdomain.lo() / source_tile_shape, target_subdomain.hi() / source_tile_shape};

  const auto& target_tile_shape = target_tiling->tile_shape_dp();
  const auto target_colors =
    Domain{source_subdomain.lo() / target_tile_shape, source_subdomain.hi() / target_tile_shape};

  const auto node_count = Runtime::get_runtime().node_count();

  SmallVector<Domain> recv_subdomains(tags::size_tag, node_count, {});
  SmallVector<Domain> send_subdomains(tags::size_tag, node_count, {});

  SmallVector<DomainPoint> recv_strides(tags::size_tag, node_count, {});
  SmallVector<DomainPoint> send_strides(tags::size_tag, node_count, {});

  SmallVector<int> recv_counts(tags::size_tag, node_count, 0);
  SmallVector<int> send_counts(tags::size_tag, node_count, 0);

  for (Domain::DomainPointIterator it{source_colors}; it; ++it) {
    const auto source_device_id = source_tiling->compute_device_id(it.p);
    const auto recv_subdomain   = source_tiling->shard(target_subdomain, it.p);

    recv_subdomains[source_device_id] = recv_subdomain;
    recv_strides[source_device_id]    = calculate_strides(recv_subdomain);
    recv_counts[source_device_id]     = recv_subdomain.get_volume();
  }

  for (Domain::DomainPointIterator it{target_colors}; it; ++it) {
    const auto target_device_id = target_tiling->compute_device_id(it.p);
    const auto send_subdomain   = target_tiling->shard(source_subdomain, it.p);

    send_subdomains[target_device_id] = send_subdomain;
    send_strides[target_device_id]    = calculate_strides(send_subdomain);
    send_counts[target_device_id]     = send_subdomain.get_volume();
  }

  InlineStorage send_buffer{source_subdomain, field_size_, source_->target()};
  InlineStorage recv_buffer{target_subdomain, field_size_, target_->target()};

  SmallVector<int> recv_offsets(tags::size_tag, node_count, 0);
  SmallVector<int> send_offsets(tags::size_tag, node_count, 0);

  for (std::uint32_t idx = 1; idx < node_count; ++idx) {
    recv_offsets[idx] = recv_offsets[idx - 1] + recv_counts[idx - 1];
    send_offsets[idx] = send_offsets[idx - 1] + send_counts[idx - 1];
  }

  LEGATE_ASSERT(
    source_subdomain.get_volume() ==
    static_cast<std::size_t>(send_offsets[node_count - 1] + send_counts[node_count - 1]));
  LEGATE_ASSERT(
    target_subdomain.get_volume() ==
    static_cast<std::size_t>(recv_offsets[node_count - 1] + recv_counts[node_count - 1]));

  cpu_copy_to_send_buffer(send_buffer,
                          *source_,
                          source_subdomain,
                          calculate_strides(source_tile_shape),
                          send_subdomains,
                          send_strides,
                          send_counts,
                          send_offsets,
                          field_size_);

  collAlltoallv(send_buffer.data(),
                send_counts.data(),
                send_offsets.data(),
                recv_buffer.data(),
                recv_counts.data(),
                recv_offsets.data(),
                to_coll_data_type_(field_size_),
                Runtime::get_runtime().get_cpu_comm());

  cpu_copy_from_recv_buffer(*target_,
                            target_subdomain,
                            calculate_strides(target_tile_shape),
                            recv_buffer,
                            recv_subdomains,
                            recv_strides,
                            recv_counts,
                            recv_offsets,
                            field_size_);
}

void CopyRequest::shuffle_(VariantCode variant_code) const
{
  if (variant_code == VariantCode::CPU) {
    cpu_shuffle_();
    return;
  }

  // If the source_partition_ was a NoPartition, the subsumption check would have passed.
  LEGATE_ASSERT(dynamic_cast<Tiling*>(source_partition_.get()));

  const auto* source_tiling = static_cast<Tiling*>(source_partition_.get());
  const auto* target_tiling = static_cast<Tiling*>(target_partition_.get());

  const auto node_id = Runtime::get_runtime().node_id();

  const auto target_subdomain =
    target_tiling->shard(sub_domain_, target_tiling->compute_color(node_id));
  const auto source_subdomain =
    source_tiling->shard(sub_domain_, source_tiling->compute_color(node_id));

  const auto& source_tile_shape = source_tiling->tile_shape_dp();
  const auto source_colors =
    Domain{target_subdomain.lo() / source_tile_shape, target_subdomain.hi() / source_tile_shape};

  const auto& target_tile_shape = target_tiling->tile_shape_dp();
  const auto target_colors =
    Domain{source_subdomain.lo() / target_tile_shape, source_subdomain.hi() / target_tile_shape};

  const auto node_count = Runtime::get_runtime().node_count();

  SmallVector<Domain> recv_subdomains(tags::size_tag, node_count, {});
  SmallVector<Domain> send_subdomains(tags::size_tag, node_count, {});

  SmallVector<DomainPoint> recv_strides(tags::size_tag, node_count, {});
  SmallVector<DomainPoint> send_strides(tags::size_tag, node_count, {});

  SmallVector<int> recv_counts(tags::size_tag, node_count, 0);
  SmallVector<int> send_counts(tags::size_tag, node_count, 0);

  for (Domain::DomainPointIterator it{source_colors}; it; ++it) {
    const auto source_device_id = source_tiling->compute_device_id(it.p);
    const auto recv_subdomain   = source_tiling->shard(target_subdomain, it.p);

    recv_subdomains[source_device_id] = recv_subdomain;
    recv_strides[source_device_id]    = calculate_strides(recv_subdomain);
    recv_counts[source_device_id]     = recv_subdomain.get_volume();
  }

  for (Domain::DomainPointIterator it{target_colors}; it; ++it) {
    const auto target_device_id = target_tiling->compute_device_id(it.p);
    const auto send_subdomain   = target_tiling->shard(source_subdomain, it.p);

    send_subdomains[target_device_id] = send_subdomain;
    send_strides[target_device_id]    = calculate_strides(send_subdomain);
    send_counts[target_device_id]     = send_subdomain.get_volume();
  }

  SmallVector<InternalSharedPtr<InlineStorage>> send_buffers(tags::size_tag, node_count, nullptr);
  SmallVector<InternalSharedPtr<InlineStorage>> recv_buffers(tags::size_tag, node_count, nullptr);

  for (auto&& [idx, send_domain] : enumerate(send_subdomains)) {
    if (send_counts[idx] == 0) {
      continue;
    }

    send_buffers[idx] =
      make_internal_shared<InlineStorage>(send_domain, field_size_, source_->target());
  }

  for (auto&& [idx, recv_domain] : enumerate(recv_subdomains)) {
    if (recv_counts[idx] == 0) {
      continue;
    }

    recv_buffers[idx] =
      make_internal_shared<InlineStorage>(recv_domain, field_size_, target_->target());
  }

  const auto stream = Runtime::get_runtime().get_cuda_stream();

  auto source_strides = calculate_strides(source_tile_shape);

  gpu_copy_to_send_buffer(send_buffers,
                          *source_,
                          source_subdomain,
                          source_strides,
                          send_subdomains,
                          send_strides,
                          send_counts,
                          field_size_,
                          stream);

  const auto comm           = Runtime::get_runtime().get_gpu_comm();
  const auto nccl_data_type = to_nccl_data_type(field_size_);

  LEGATE_CHECK_NCCL(ncclGroupStart());

  for (auto&& [idx, send_domain] : enumerate(send_subdomains)) {
    if (send_counts[idx] == 0) {
      continue;
    }

    LEGATE_CHECK_NCCL(
      ncclSend(send_buffers[idx]->data(), send_counts[idx], nccl_data_type, idx, comm, stream));
  }

  for (auto&& [idx, recv_domain] : enumerate(recv_subdomains)) {
    if (recv_counts[idx] == 0) {
      continue;
    }

    LEGATE_CHECK_NCCL(
      ncclRecv(recv_buffers[idx]->data(), recv_counts[idx], nccl_data_type, idx, comm, stream));
  }

  LEGATE_CHECK_NCCL(ncclGroupEnd());

  auto target_strides = calculate_strides(target_tile_shape);

  gpu_copy_from_recv_buffer(*target_,
                            target_subdomain,
                            target_strides,
                            recv_buffers,
                            recv_subdomains,
                            recv_strides,
                            recv_counts,
                            field_size_,
                            stream);
}

}  // namespace legate::detail
