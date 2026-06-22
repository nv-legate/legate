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
#include <tuple>

namespace legate::detail {

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

Logger& log_copy()
{
  static Logger log{"legate.copy"};

  return log;
}

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) field_size_dispatch(std::uint32_t field_size, Functor f, Fnargs&&... args)
{
  switch (field_size) {
    case 1: {
      return f.template operator()<std::uint8_t>(std::forward<Fnargs>(args)...);
    }
    case 2: {
      return f.template operator()<std::uint16_t>(std::forward<Fnargs>(args)...);
    }
    case 4: {
      return f.template operator()<std::uint32_t>(std::forward<Fnargs>(args)...);
    }
    case 8: {
      return f.template operator()<std::uint64_t>(std::forward<Fnargs>(args)...);
    }
    default: {  // legate-lint: no-switch-default
      throw TracedException<std::runtime_error>{"Unsupported field size"};
    }
  }
  LEGATE_UNREACHABLE();
  return f.template operator()<std::uint8_t>(std::forward<Fnargs>(args)...);
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

struct CopyToSendBufferCPU {
  template <typename VAL>
  void operator()(InlineStorage& target,
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
      }
    }
  }
};

void copy_to_send_buffer_cpu_(std::uint32_t field_size,
                              InlineStorage& target,
                              const InlineStorage& source,
                              const Domain& source_domain,
                              const DomainPoint& source_strides,
                              const SmallVector<Domain>& target_subdomains,
                              const SmallVector<DomainPoint>& all_target_strides,
                              const SmallVector<int>& counts,
                              const SmallVector<int>& offsets)
{
  field_size_dispatch(field_size,
                      CopyToSendBufferCPU{},
                      target,
                      source,
                      source_domain,
                      source_strides,
                      target_subdomains,
                      all_target_strides,
                      counts,
                      offsets);
}

struct CopyFromRecvBufferCPU {
  template <typename VAL>
  void operator()(InlineStorage& target,
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
};

void copy_from_recv_buffer_cpu_(std::uint32_t field_size,
                                InlineStorage& target,
                                const Domain& target_domain,
                                const DomainPoint& target_strides,
                                const InlineStorage& source,
                                const SmallVector<Domain>& source_subdomains,
                                const SmallVector<DomainPoint>& all_source_strides,
                                const SmallVector<int>& counts,
                                const SmallVector<int>& offsets)
{
  field_size_dispatch(field_size,
                      CopyFromRecvBufferCPU{},
                      target,
                      target_domain,
                      target_strides,
                      source,
                      source_subdomains,
                      all_source_strides,
                      counts,
                      offsets);
}

template <typename VAL>
struct CopyToSendBufferKernelName;

template <>
struct CopyToSendBufferKernelName<std::uint8_t> {
  static constexpr const char* value = "legate_copy_subdomain_8";
};

template <>
struct CopyToSendBufferKernelName<std::uint16_t> {
  static constexpr const char* value = "legate_copy_subdomain_16";
};

template <>
struct CopyToSendBufferKernelName<std::uint32_t> {
  static constexpr const char* value = "legate_copy_subdomain_32";
};

template <>
struct CopyToSendBufferKernelName<std::uint64_t> {
  static constexpr const char* value = "legate_copy_subdomain_64";
};

struct CopyToSendBufferGPU {
  template <typename VAL>
  void operator()(SmallVector<InternalSharedPtr<InlineStorage>>& targets,
                  const InlineStorage& source,
                  const Domain& source_domain,
                  DomainPoint& source_strides,
                  const SmallVector<Domain>& target_subdomains,
                  SmallVector<DomainPoint>& all_target_strides,
                  const SmallVector<int>& counts,
                  CUstream stream)
  {
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

    auto&& api         = cuda::detail::get_cuda_driver_api();
    auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();
    CUkernel kern = mod_manager.load_kernel_from_fatbin(copy_request_fatbin,
                                                        CopyToSendBufferKernelName<VAL>::value);

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

      const auto num_blocks =
        (static_cast<std::size_t>(counts[idx]) + LEGATE_THREADS_PER_BLOCK - 1) /
        LEGATE_THREADS_PER_BLOCK;
      auto count          = counts[idx];
      auto target_extents = calculate_extents(target_subdomain);

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
    }
  }
};

void copy_to_send_buffer_gpu_(std::uint32_t field_size,
                              SmallVector<InternalSharedPtr<InlineStorage>>& targets,
                              const InlineStorage& source,
                              const Domain& source_domain,
                              DomainPoint& source_strides,
                              const SmallVector<Domain>& target_subdomains,
                              SmallVector<DomainPoint>& all_target_strides,
                              const SmallVector<int>& counts,
                              CUstream stream)
{
  field_size_dispatch(field_size,
                      CopyToSendBufferGPU{},
                      targets,
                      source,
                      source_domain,
                      source_strides,
                      target_subdomains,
                      all_target_strides,
                      counts,
                      stream);
}

struct CopyFromRecvBufferGPU {
  template <typename VAL>
  void operator()(InlineStorage& target,
                  const Domain& target_domain,
                  DomainPoint& target_strides,
                  const SmallVector<InternalSharedPtr<InlineStorage>>& sources,
                  const SmallVector<Domain>& source_subdomains,
                  SmallVector<DomainPoint>& all_source_strides,
                  const SmallVector<int>& counts,
                  CUstream stream)
  {
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

    auto&& api         = cuda::detail::get_cuda_driver_api();
    auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();
    CUkernel kern = mod_manager.load_kernel_from_fatbin(copy_request_fatbin,
                                                        CopyToSendBufferKernelName<VAL>::value);

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

      const auto num_blocks =
        (static_cast<std::size_t>(counts[idx]) + LEGATE_THREADS_PER_BLOCK - 1) /
        LEGATE_THREADS_PER_BLOCK;
      auto count          = counts[idx];
      auto source_extents = calculate_extents(source_subdomain);

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
};

void copy_from_recv_buffer_gpu_(std::uint32_t field_size,
                                InlineStorage& target,
                                const Domain& target_domain,
                                DomainPoint& target_strides,
                                const SmallVector<InternalSharedPtr<InlineStorage>>& sources,
                                const SmallVector<Domain>& source_subdomains,
                                SmallVector<DomainPoint>& all_source_strides,
                                const SmallVector<int>& counts,
                                CUstream stream)
{
  field_size_dispatch(field_size,
                      CopyFromRecvBufferGPU{},
                      target,
                      target_domain,
                      target_strides,
                      sources,
                      source_subdomains,
                      all_source_strides,
                      counts,
                      stream);
}

std::tuple<Domain, SmallVector<Domain>, SmallVector<DomainPoint>, SmallVector<int>>
find_all_intersections(const Tiling* from, const Tiling* to, const Domain& store_domain)
{
  const auto device_count = Runtime::get_runtime().node_count();
  const auto my_device_id = Runtime::get_runtime().node_id();

  const auto from_tile_domain    = from->shard(store_domain, from->compute_color(my_device_id));
  const auto intersection_colors = Domain{from_tile_domain.lo() / to->tile_shape_dp(),
                                          from_tile_domain.hi() / to->tile_shape_dp()};

  auto intersection_subdomains = SmallVector<Domain>(tags::size_tag, device_count, {});
  auto intersection_strides    = SmallVector<DomainPoint>(tags::size_tag, device_count, {});
  auto intersection_counts     = SmallVector<int>(tags::size_tag, device_count, {});

  for (Domain::DomainPointIterator it{intersection_colors}; it; ++it) {
    const auto to_device_id = to->compute_device_id(it.p);
    const auto subdomain    = to->shard(from_tile_domain, it.p);

    intersection_subdomains[to_device_id] = subdomain;
    intersection_strides[to_device_id]    = calculate_strides(subdomain);
    intersection_counts[to_device_id]     = subdomain.get_volume();
  }

  return {from_tile_domain, intersection_subdomains, intersection_strides, intersection_counts};
}

}  // namespace

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

void CopyRequest::shuffle_(VariantCode variant_code) const
{
  // If the source_partition_ was a NoPartition, the subsumption check would have passed.
  LEGATE_ASSERT(dynamic_cast<Tiling*>(source_partition_.get()));

  const auto* source_tiling = static_cast<Tiling*>(source_partition_.get());
  const auto* target_tiling = static_cast<Tiling*>(target_partition_.get());
  const auto device_count   = Runtime::get_runtime().node_count();

  auto&& [source_subdomain, send_subdomains, send_strides, send_counts] =
    find_all_intersections(source_tiling, target_tiling, sub_domain_);
  auto&& [target_subdomain, recv_subdomains, recv_strides, recv_counts] =
    find_all_intersections(target_tiling, source_tiling, sub_domain_);

  // TODO(wonchanl): Add an option to toggle this behavior
  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    std::stringstream ss{};

    ss << "SHUFFLE\n  from: " << source_tiling->to_string()
       << "\n  to: " << target_tiling->to_string() << "\n  store_domain: " << sub_domain_
       << "\n  intersections:\n";
    for (auto&& [to, send_domain] : enumerate(send_subdomains)) {
      if (send_counts[to] == 0) {
        continue;
      }
      ss << "    send to " << to << ": " << send_domain << "\n";
    }
    for (auto&& [from, recv_domain] : enumerate(recv_subdomains)) {
      if (recv_counts[from] == 0) {
        continue;
      }
      ss << "    recv from " << from << ": " << recv_domain << "\n";
    }
    log_copy().debug() << std::move(ss).str();
  }

  if (variant_code == VariantCode::CPU) {
    // For CPU shuffle, we allocate contiguous transfer buffers to match the all-to-all API
    InlineStorage send_buffer{source_subdomain, field_size_, source_->target()};
    InlineStorage recv_buffer{target_subdomain, field_size_, target_->target()};

    SmallVector<int> recv_offsets(tags::size_tag, device_count, 0);
    SmallVector<int> send_offsets(tags::size_tag, device_count, 0);

    for (std::uint32_t idx = 1; idx < device_count; ++idx) {
      recv_offsets[idx] = recv_offsets[idx - 1] + recv_counts[idx - 1];
      send_offsets[idx] = send_offsets[idx - 1] + send_counts[idx - 1];
    }

    LEGATE_ASSERT(
      source_subdomain.get_volume() ==
      static_cast<std::size_t>(send_offsets[device_count - 1] + send_counts[device_count - 1]));
    LEGATE_ASSERT(
      target_subdomain.get_volume() ==
      static_cast<std::size_t>(recv_offsets[device_count - 1] + recv_counts[device_count - 1]));

    copy_to_send_buffer_cpu_(field_size_,
                             send_buffer,
                             *source_,
                             source_subdomain,
                             calculate_strides(source_tiling->tile_shape_dp()),
                             send_subdomains,
                             send_strides,
                             send_counts,
                             send_offsets);

    collAlltoallv(send_buffer.data(),
                  send_counts.data(),
                  send_offsets.data(),
                  recv_buffer.data(),
                  recv_counts.data(),
                  recv_offsets.data(),
                  to_coll_data_type_(field_size_),
                  Runtime::get_runtime().get_cpu_comm());

    copy_from_recv_buffer_cpu_(field_size_,
                               *target_,
                               target_subdomain,
                               calculate_strides(target_tiling->tile_shape_dp()),
                               recv_buffer,
                               recv_subdomains,
                               recv_strides,
                               recv_counts,
                               recv_offsets);

    return;
  }

  LEGATE_ASSERT(variant_code == VariantCode::GPU);

  // For GPU shuffle, we allocate a separate buffer for each intersection
  SmallVector<InternalSharedPtr<InlineStorage>> send_buffers(tags::size_tag, device_count, nullptr);
  SmallVector<InternalSharedPtr<InlineStorage>> recv_buffers(tags::size_tag, device_count, nullptr);

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

  const auto stream         = Runtime::get_runtime().get_cuda_stream();
  const auto comm           = Runtime::get_runtime().get_gpu_comm();
  const auto nccl_data_type = to_nccl_data_type(field_size_);

  auto source_strides = calculate_strides(source_tiling->tile_shape_dp());
  auto target_strides = calculate_strides(target_tiling->tile_shape_dp());

  copy_to_send_buffer_gpu_(field_size_,
                           send_buffers,
                           *source_,
                           source_subdomain,
                           source_strides,
                           send_subdomains,
                           send_strides,
                           send_counts,
                           stream);

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

  copy_from_recv_buffer_gpu_(field_size_,
                             *target_,
                             target_subdomain,
                             target_strides,
                             recv_buffers,
                             recv_subdomains,
                             recv_strides,
                             recv_counts,
                             stream);
}

}  // namespace legate::detail
