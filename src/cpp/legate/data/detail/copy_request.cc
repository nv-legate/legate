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
#include <legate/geometry/detail/device_domain_point.h>
#include <legate/geometry/domain.h>
#include <legate/geometry/domain_point.h>
#include <legate/partitioning/detail/partition/no_partition.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/array_algorithms.h>
#include <legate/utilities/detail/divmod.h>
#include <legate/utilities/detail/enumerate.h>
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

// Converts a DomainPoint into a trivially-copyable POD that can be passed by value into a CUDA
// kernel (the SmallVector-backed DomainPoint cannot be memcpy'd into kernel parameter space).
[[nodiscard]] DeviceDomainPoint to_device_domain_point(const DomainPoint& point)
{
  DeviceDomainPoint result;

  result.dim = point.dim();
  for (std::uint32_t idx = 0; idx < point.dim(); ++idx) {
    result[idx] = point[idx];
  }
  return result;
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

  const auto target_strides = target.domain().strides();

  const auto& tile_shape  = tiling->tile_shape();
  const auto tile_strides = tile_shape.strides();
  const auto tile_volume  = tile_shape.volume();

  const auto color_strides = tiling->color_shape().strides();

  const auto target_lo = target.domain().lo();

  for (auto&& point : target.domain().iterate()) {
    // `target_idx` is relative to the target's origin, whereas the tile-major `source_idx` is
    // computed from the global coordinate `point` (the tiling lives in the storage's global
    // coordinate space). These differ whenever the target is offset (e.g. a slice).
    const auto target_idx = (point - target_lo).dot(target_strides);
    const auto source_idx = (point % tile_shape).dot(tile_strides) +
                            (point / tile_shape).dot(color_strides) * tile_volume;

    target_ptr[target_idx] = source_ptr[source_idx];
  }
}

struct ReorderElementsCPU {
  template <typename VAL>
  void operator()(InlineStorage& target, const InlineStorage& source, const Tiling* tiling)
  {
    cpu_reorder_elements_<VAL>(target, source, tiling);
  }
};

void reorder_elements_cpu_(std::uint32_t field_size,
                           InlineStorage& target,
                           const InlineStorage& source,
                           const Tiling* tiling)
{
  field_size_dispatch(field_size, ReorderElementsCPU{}, target, source, tiling);
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
      static_cast<const VAL*>(source.data()) - source_domain.lo().dot(source_strides);

    for (std::uint32_t idx = 0; idx < node_count; ++idx) {
      if (counts[idx] == 0) {
        continue;
      }

      const auto& target_subdomain = target_subdomains[idx];
      const auto& target_strides   = all_target_strides[idx];
      auto* target_ptr =
        static_cast<VAL*>(target.data()) + offsets[idx] - target_subdomain.lo().dot(target_strides);

      for (auto&& point : target_subdomain.iterate()) {
        target_ptr[point.dot(target_strides)] = source_ptr[point.dot(source_strides)];
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

    auto* target_ptr = static_cast<VAL*>(target.data()) - target_domain.lo().dot(target_strides);

    for (std::uint32_t idx = 0; idx < node_count; ++idx) {
      if (counts[idx] == 0) {
        continue;
      }

      const auto& source_subdomain = source_subdomains[idx];
      const auto& source_strides   = all_source_strides[idx];
      const auto* source_ptr       = static_cast<const VAL*>(source.data()) + offsets[idx] -
                                     source_subdomain.lo().dot(source_strides);

      for (auto&& point : source_subdomain.iterate()) {
        target_ptr[point.dot(target_strides)] = source_ptr[point.dot(source_strides)];
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
  static constexpr const char* value = "legate_copy_to_send_buffer_8";
};

template <>
struct CopyToSendBufferKernelName<std::uint16_t> {
  static constexpr const char* value = "legate_copy_to_send_buffer_16";
};

template <>
struct CopyToSendBufferKernelName<std::uint32_t> {
  static constexpr const char* value = "legate_copy_to_send_buffer_32";
};

template <>
struct CopyToSendBufferKernelName<std::uint64_t> {
  static constexpr const char* value = "legate_copy_to_send_buffer_64";
};

struct CopyToSendBufferGPU {
  template <typename VAL>
  void operator()(SmallVector<std::pair<InternalSharedPtr<InlineStorage>, bool>>& targets,
                  const InlineStorage& source,
                  const Domain& source_domain,
                  const DomainPoint& source_strides,
                  const SmallVector<Domain>& target_subdomains,
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
      static_cast<const VAL*>(source.data()) - source_domain.lo().dot(source_strides);

    for (std::uint32_t idx = 0; idx < node_count; ++idx) {
      if (counts[idx] == 0) {
        continue;
      }

      auto&& [target, needs_copy] = targets[idx];

      if (!needs_copy) {
        continue;
      }

      const auto& target_subdomain   = target_subdomains[idx];
      auto* target_ptr               = static_cast<VAL*>(target->data());
      const auto* source_ptr_shifted = source_ptr + target_subdomain.lo().dot(source_strides);

      const auto num_blocks =
        (static_cast<std::size_t>(counts[idx]) + LEGATE_THREADS_PER_BLOCK - 1) /
        LEGATE_THREADS_PER_BLOCK;
      auto count          = counts[idx];
      auto target_extents = target_subdomain.extents();

      if (source_domain.dim() == 2) {
        api->mem_cpy_2d_async(target_ptr,
                              source_ptr_shifted,
                              target_extents[1] * sizeof(VAL),
                              source_strides[0] * sizeof(VAL),
                              target_extents[1] * sizeof(VAL),
                              target_extents[0],
                              stream);
        continue;
      }

      FastDivmodU64 divmod{source_domain.dim() == 2 ? static_cast<std::uint64_t>(target_extents[1])
                                                    : 1};

      api->launch_kernel(kern,
                         {num_blocks},
                         {LEGATE_THREADS_PER_BLOCK},
                         /*shared_mem_bytes=*/0,
                         stream,
                         static_cast<void*>(target_ptr),
                         static_cast<const void*>(source_ptr_shifted),
                         count,
                         to_device_domain_point(target_extents),
                         divmod,
                         to_device_domain_point(source_strides));
    }
  }
};

void copy_to_send_buffer_gpu_(
  std::uint32_t field_size,
  SmallVector<std::pair<InternalSharedPtr<InlineStorage>, bool>>& targets,
  const InlineStorage& source,
  const Domain& source_domain,
  const DomainPoint& source_strides,
  const SmallVector<Domain>& target_subdomains,
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
                      counts,
                      stream);
}

template <typename VAL>
struct CopyFromRecvBufferKernelName;

template <>
struct CopyFromRecvBufferKernelName<std::uint8_t> {
  static constexpr const char* value = "legate_copy_from_recv_buffer_8";
};

template <>
struct CopyFromRecvBufferKernelName<std::uint16_t> {
  static constexpr const char* value = "legate_copy_from_recv_buffer_16";
};

template <>
struct CopyFromRecvBufferKernelName<std::uint32_t> {
  static constexpr const char* value = "legate_copy_from_recv_buffer_32";
};

template <>
struct CopyFromRecvBufferKernelName<std::uint64_t> {
  static constexpr const char* value = "legate_copy_from_recv_buffer_64";
};

struct CopyFromRecvBufferGPU {
  template <typename VAL>
  void operator()(InlineStorage& target,
                  const Domain& target_domain,
                  const DomainPoint& target_strides,
                  const SmallVector<std::pair<InternalSharedPtr<InlineStorage>, bool>>& sources,
                  const SmallVector<Domain>& source_subdomains,
                  const SmallVector<int>& counts,
                  CUstream stream)
  {
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};
    auto&& api            = cuda::detail::get_cuda_driver_api();
    const auto node_count = Runtime::get_runtime().node_count();

    auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();
    CUkernel kern = mod_manager.load_kernel_from_fatbin(copy_request_fatbin,
                                                        CopyFromRecvBufferKernelName<VAL>::value);

    auto* target_ptr = static_cast<VAL*>(target.data()) - target_domain.lo().dot(target_strides);

    for (std::uint32_t idx = 0; idx < node_count; ++idx) {
      if (counts[idx] == 0) {
        continue;
      }

      auto&& [source, needs_copy] = sources[idx];

      if (!needs_copy) {
        continue;
      }

      const auto& source_subdomain = source_subdomains[idx];
      auto* target_ptr_shifted     = target_ptr + source_subdomain.lo().dot(target_strides);
      const auto* source_ptr       = static_cast<const VAL*>(source->data());

      const auto num_blocks =
        (static_cast<std::size_t>(counts[idx]) + LEGATE_THREADS_PER_BLOCK - 1) /
        LEGATE_THREADS_PER_BLOCK;
      auto count          = counts[idx];
      auto source_extents = source_subdomain.extents();

      if (target_domain.dim() == 2) {
        api->mem_cpy_2d_async(target_ptr_shifted,
                              source_ptr,
                              target_strides[0] * sizeof(VAL),
                              source_extents[1] * sizeof(VAL),
                              source_extents[1] * sizeof(VAL),
                              source_extents[0],
                              stream);
        continue;
      }

      FastDivmodU64 divmod{target_domain.dim() == 2 ? static_cast<std::uint64_t>(source_extents[1])
                                                    : 1};

      api->launch_kernel(kern,
                         {num_blocks},
                         {LEGATE_THREADS_PER_BLOCK},
                         /*shared_mem_bytes=*/0,
                         stream,
                         static_cast<void*>(target_ptr_shifted),
                         static_cast<const void*>(source_ptr),
                         count,
                         to_device_domain_point(source_extents),
                         divmod,
                         to_device_domain_point(target_strides));
    }
  }
};

void copy_from_recv_buffer_gpu_(
  std::uint32_t field_size,
  InlineStorage& target,
  const Domain& target_domain,
  const DomainPoint& target_strides,
  const SmallVector<std::pair<InternalSharedPtr<InlineStorage>, bool>>& sources,
  const SmallVector<Domain>& source_subdomains,
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
                      counts,
                      stream);
}

template <typename VAL>
struct ReorderElementsKernelName;

template <>
struct ReorderElementsKernelName<std::uint8_t> {
  static constexpr const char* value = "legate_reorder_elements_8";
};

template <>
struct ReorderElementsKernelName<std::uint16_t> {
  static constexpr const char* value = "legate_reorder_elements_16";
};

template <>
struct ReorderElementsKernelName<std::uint32_t> {
  static constexpr const char* value = "legate_reorder_elements_32";
};

template <>
struct ReorderElementsKernelName<std::uint64_t> {
  static constexpr const char* value = "legate_reorder_elements_64";
};

struct ReorderElementsGPU {
  template <typename VAL>
  void operator()(InlineStorage& target,
                  const InlineStorage& source,
                  const Tiling* tiling,
                  CUstream stream)
  {
    auto volume = target.domain().volume();

    if (volume == 0) {
      return;
    }

    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};
    auto&& api         = cuda::detail::get_cuda_driver_api();
    auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();
    CUkernel kern      = mod_manager.load_kernel_from_fatbin(copy_request_fatbin,
                                                             ReorderElementsKernelName<VAL>::value);

    auto target_extents      = target.domain().extents();
    auto target_lo           = target.domain().lo();
    auto tile_shape          = tiling->tile_shape();
    auto tile_strides        = tile_shape.strides();
    auto color_strides       = tiling->color_shape().strides();
    std::int64_t tile_volume = tile_shape.volume();

    const auto num_blocks = (volume + LEGATE_THREADS_PER_BLOCK - 1) / LEGATE_THREADS_PER_BLOCK;

    api->launch_kernel(kern,
                       {num_blocks},
                       {LEGATE_THREADS_PER_BLOCK},
                       /*shared_mem_bytes=*/0,
                       stream,
                       target.data(),
                       source.data(),
                       volume,
                       to_device_domain_point(target_extents),
                       to_device_domain_point(target_lo),
                       to_device_domain_point(tile_shape),
                       to_device_domain_point(tile_strides),
                       to_device_domain_point(color_strides),
                       tile_volume);
  }
};

void reorder_elements_gpu_(std::uint32_t field_size,
                           InlineStorage& target,
                           const InlineStorage& source,
                           const Tiling* tiling,
                           CUstream stream)
{
  field_size_dispatch(field_size, ReorderElementsGPU{}, target, source, tiling, stream);
}

// Computes, for this device's `from` piece, the overlapping sub-domains with every device's `to`
// piece, indexed by the destination device id. Works for any pair of partitions (`Tiling`,
// `Opaque`, or `NoPartition`): the `from`/`to` pieces are obtained through the polymorphic
// `compute_color`/`shard`/`compute_device_ids` interface and the color space is enumerated via
// `launch_domain()`. A partition with no color space (`NoPartition`) is treated as a single piece
// covering the whole store on every device.
std::tuple<Domain, SmallVector<Domain>, SmallVector<DomainPoint>, SmallVector<int>>
find_all_intersections(const Partition* from,
                       const Partition* to,
                       const Domain& from_domain,
                       const Domain& to_domain)
{
  const auto device_count = Runtime::get_runtime().node_count();
  const auto my_device_id = Runtime::get_runtime().node_id();

  auto intersection_subdomains = SmallVector<Domain>(tags::size_tag, device_count, {});
  auto intersection_strides    = SmallVector<DomainPoint>(tags::size_tag, device_count, {});
  auto intersection_counts     = SmallVector<int>(tags::size_tag, device_count, {});

  // This device's `from` piece, clipped to the `to` store's domain.
  Domain from_tile_domain;

  if (from->has_color_shape()) {
    // A device that the `from` partition does not map to (e.g. one outside a partially-replicated
    // mesh) owns no piece and has nothing to exchange.
    const auto my_color = from->compute_color(my_device_id);

    if (my_color.dim() == 0) {
      return {Domain{}, intersection_subdomains, intersection_strides, intersection_counts};
    }
    from_tile_domain = from->shard(from_domain, my_color).intersection(to_domain);
  } else {
    // No color space: the whole store lives on this device.
    from_tile_domain = from_domain.intersection(to_domain);
  }

  if (from_tile_domain.volume() == 0) {
    return {from_tile_domain, intersection_subdomains, intersection_strides, intersection_counts};
  }

  // Records the overlap of `from_tile_domain` with the `to` piece of a given color. A destination
  // color may be held by more than one device when `to` replicates the piece; the same subdomain is
  // exchanged with each of them.
  const auto record = [&](const DomainPoint& color, const Domain& piece) {
    const auto subdomain = piece.intersection(from_tile_domain);

    if (subdomain.volume() == 0) {
      return;
    }

    const auto strides = subdomain.strides();
    const auto count   = static_cast<int>(subdomain.volume());

    for (const auto to_device_id : to->compute_device_ids(color)) {
      intersection_subdomains[to_device_id] = subdomain;
      intersection_strides[to_device_id]    = strides;
      intersection_counts[to_device_id]     = count;
    }
  };

  if (to->has_color_shape()) {
    for (auto&& point : to->launch_domain().iterate()) {
      record(point, to->shard(to_domain, point));
    }
  } else {
    // No color space: a single piece (the whole store) held on every device.
    record(DomainPoint{}, to_domain);
  }

  return {from_tile_domain, intersection_subdomains, intersection_strides, intersection_counts};
}

}  // namespace

void CopyRequest::perform(VariantCode variant_code) const
{
  LEGATE_CHECK(source_partition_->has_color_shape());

  if (source_partition_->is_equi_size(source_domain_) &&
      (!target_partition_->has_color_shape() || (target_partition_->color_shape().volume() == 1))) {
    all_gather_(variant_code);
  } else {
    shuffle_(variant_code);
  }
}

void CopyRequest::all_gather_(VariantCode variant_code) const
{
  // TODO(wonchanl): Add an option to toggle this behavior
  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    std::stringstream ss{};

    ss << "ALL_GATHER(storage_id: " << storage_id_ << ")\n";
    ss << "  from: " << source_partition_->to_string() << "\n";
    ss << "    domain: " << source_domain_ << "\n";
    ss << "  to: " << target_partition_->to_string() << "\n";
    ss << "    domain: " << target_domain_ << "\n";
    log_copy().debug() << std::move(ss).str();
  }

  // TODO(wonchanl): The code below doesn't handle the case where there are smaller tiles

  // All-gathers the source into `dest` using the collective backing the given variant.
  const auto all_gather_fn = [this, variant_code](InlineStorage& dest) {
    if (variant_code == VariantCode::CPU) {
      collAllgather(source_->data(),
                    dest.data(),
                    static_cast<int>(source_->domain().volume()),
                    to_coll_data_type_(field_size_),
                    Runtime::get_runtime().get_cpu_comm());
    } else {
      LEGATE_CHECK(variant_code == VariantCode::GPU);
      LEGATE_CHECK_NCCL(ncclAllGather(source_->data(),
                                      dest.data(),
                                      source_->domain().volume(),
                                      to_nccl_data_type(field_size_),
                                      Runtime::get_runtime().get_gpu_comm(),
                                      Runtime::get_runtime().get_cuda_stream()));
    }
  };

  // The all-gathered result is laid out tile-major: one source tile per participating device,
  // concatenated in device order. The gathered buffer therefore holds the *total* gathered volume
  // rather than the (possibly smaller and/or offset) target domain.
  //
  // FIXME(wonchan): source_->domain() may not be the tile shape if source_ isn't a tight
  // allocation
  const auto gathered_volume = static_cast<std::int64_t>(Runtime::get_runtime().node_count()) *
                               static_cast<std::int64_t>(source_->domain().volume());

  // We can all-gather directly into the target only when the tile-major layout produced by the
  // collective already coincides with the target's row-major layout. That is the case iff all of:
  //
  //   1. The tiles are stacked along the first dimension only (color shape == (P, 1, ..., 1)), so
  //      concatenating tiles in device order yields exactly a row-major buffer. Checking only the
  //      last color dimension is insufficient: e.g. a (1, P) color is contiguous per-tile but not
  //      row-major across tiles.
  //   2. The target starts at the storage origin (no slice offset). A sliced target has its data
  //      at non-zero global coordinates, so a straight copy would land the elements in the wrong
  //      place.
  //   3. The target exactly holds the whole gathered region (its volume equals the total gathered
  //      volume). Otherwise the direct all-gather would overflow the target buffer (when the
  //      gathered region is larger, e.g. because of padded/partial tiles) or leave it only
  //      partially filled.
  //
  // When any of these fail, we all-gather into a temporary and reorder the elements into the target
  // below.
  //
  // TODO(wonchan): need to check if the device mesh has C order as well
  const auto& color_shape      = source_partition_->color_shape();
  auto stacked_along_first_dim = true;
  for (std::size_t dim = 1; dim < color_shape.size(); ++dim) {
    stacked_along_first_dim = stacked_along_first_dim && (color_shape[dim] == 1);
  }

  const auto target_lo  = target_->domain().lo();
  auto target_at_origin = true;
  for (std::uint32_t dim = 0; dim < target_lo.dim(); ++dim) {
    target_at_origin = target_at_origin && (target_lo[dim] == 0);
  }

  if (stacked_along_first_dim && target_at_origin &&
      target_->domain().volume() == static_cast<std::size_t>(gathered_volume)) {
    all_gather_fn(*target_);
    return;
  }

  InlineStorage tmp{Domain{Rect<1>{0, gathered_volume - 1}}, field_size_, target_->target()};

  all_gather_fn(tmp);

  // If we're here, the target partition is NoPartition, so the source partition cannot be
  // NoPartition as well.
  const auto* tiling = static_cast<Tiling*>(source_partition_.get());

  if (variant_code == VariantCode::CPU) {
    reorder_elements_cpu_(field_size_, *target_, tmp, tiling);
  } else {
    LEGATE_CHECK(variant_code == VariantCode::GPU);
    reorder_elements_gpu_(
      field_size_, *target_, tmp, tiling, Runtime::get_runtime().get_cuda_stream());
  }
}

void CopyRequest::shuffle_(VariantCode variant_code) const
{
  // The source partition always has a color shape (a NoPartition source would have been resolved by
  // the subsumption check). Both partitions are handled polymorphically, so this works for any of
  // Tiling/Opaque as source and Tiling/Opaque/NoPartition as target.
  LEGATE_ASSERT(source_partition_->has_color_shape());

  const auto* source      = source_partition_.get();
  const auto* target      = target_partition_.get();
  const auto device_count = Runtime::get_runtime().node_count();

  auto&& [send_domain, send_subdomains, send_strides, send_counts] =
    find_all_intersections(source, target, source_domain_, target_domain_);
  auto&& [recv_domain, recv_subdomains, recv_strides, recv_counts] =
    find_all_intersections(target, source, target_domain_, source_domain_);

  const auto& source_domain = source_->domain();
  const auto& target_domain = target_->domain();
  const auto source_strides = source_domain.strides();
  const auto target_strides = target_domain.strides();

  // TODO(wonchanl): Add an option to toggle this behavior
  if constexpr (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    std::stringstream ss{};

    ss << "SHUFFLE(storage_id: " << storage_id_ << ")\n";
    ss << "  from: " << source->to_string() << "\n";
    ss << "    domain: " << source_domain_ << "\n";
    ss << "  to: " << target->to_string() << "\n";
    ss << "    domain: " << target_domain_ << "\n";
    ss << "  intersections:\n";
    for (auto&& [to, send_subdomain] : enumerate(send_subdomains)) {
      if (send_counts[to] == 0) {
        continue;
      }
      ss << "    send to " << to << ": " << send_subdomain << "\n";
    }
    for (auto&& [from, recv_subdomain] : enumerate(recv_subdomains)) {
      if (recv_counts[from] == 0) {
        continue;
      }
      ss << "    recv from " << from << ": " << recv_subdomain << "\n";
    }
    log_copy().debug() << std::move(ss).str();
  }

  if (variant_code == VariantCode::CPU) {
    SmallVector<int> recv_offsets(tags::size_tag, device_count, 0);
    SmallVector<int> send_offsets(tags::size_tag, device_count, 0);

    for (std::uint32_t idx = 1; idx < device_count; ++idx) {
      recv_offsets[idx] = recv_offsets[idx - 1] + recv_counts[idx - 1];
      send_offsets[idx] = send_offsets[idx - 1] + send_counts[idx - 1];
    }

    const auto send_total = send_offsets[device_count - 1] + send_counts[device_count - 1];
    const auto recv_total = recv_offsets[device_count - 1] + recv_counts[device_count - 1];

    // The receive side always tiles the target piece disjointly, so its total equals that piece.
    LEGATE_ASSERT(recv_domain.volume() == static_cast<std::size_t>(recv_total));

    // For CPU shuffle, we allocate contiguous transfer buffers to match the all-to-all API. The
    // buffers are laid out as one chunk per peer (prefix-sum offsets). When a source piece is
    // broadcast to several peers (e.g. gathering an Opaque partition into a replicated target) the
    // same elements are packed once per destination, so the send buffer is sized by the total
    // element count rather than the (single-piece) source domain.
    InlineStorage send_buffer{
      Domain{Rect<1>{0, std::max(send_total, 1) - 1}}, field_size_, source_->target()};
    InlineStorage recv_buffer{
      Domain{Rect<1>{0, std::max(recv_total, 1) - 1}}, field_size_, target_->target()};

    copy_to_send_buffer_cpu_(field_size_,
                             send_buffer,
                             *source_,
                             source_domain,
                             source_strides,
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
                               target_domain,
                               target_strides,
                               recv_buffer,
                               recv_subdomains,
                               recv_strides,
                               recv_counts,
                               recv_offsets);

    return;
  }

  LEGATE_ASSERT(variant_code == VariantCode::GPU);

  // For GPU shuffle, we allocate a separate buffer for each intersection
  SmallVector<std::pair<InternalSharedPtr<InlineStorage>, bool>> send_buffers(
    tags::size_tag, device_count, {nullptr, false});
  SmallVector<std::pair<InternalSharedPtr<InlineStorage>, bool>> recv_buffers(
    tags::size_tag, device_count, {nullptr, false});

  for (auto&& [idx, send_subdomain] : enumerate(send_subdomains)) {
    if (send_counts[idx] == 0) {
      continue;
    }

    if (source_strides == send_subdomain.strides()) {
      send_buffers[idx] = std::make_pair<>(source_, false);
    } else {
      send_buffers[idx] = std::make_pair<>(
        make_internal_shared<InlineStorage>(send_subdomain, field_size_, source_->target()), true);
    }
  }

  for (auto&& [idx, recv_subdomain] : enumerate(recv_subdomains)) {
    if (recv_counts[idx] == 0) {
      continue;
    }

    if (target_strides == recv_subdomain.strides()) {
      recv_buffers[idx] = std::make_pair<>(target_, false);
    } else {
      recv_buffers[idx] = std::make_pair<>(
        make_internal_shared<InlineStorage>(recv_subdomain, field_size_, target_->target()), true);
    }
  }

  const auto stream         = Runtime::get_runtime().get_cuda_stream();
  const auto comm           = Runtime::get_runtime().get_gpu_comm();
  const auto nccl_data_type = to_nccl_data_type(field_size_);

  copy_to_send_buffer_gpu_(field_size_,
                           send_buffers,
                           *source_,
                           source_domain,
                           source_strides,
                           send_subdomains,
                           send_counts,
                           stream);

  LEGATE_CHECK_NCCL(ncclGroupStart());

  for (auto&& [idx, send_subdomain] : enumerate(send_subdomains)) {
    if (send_counts[idx] == 0) {
      continue;
    }

    auto&& [send_buffer, needs_copy] = send_buffers[idx];

    const auto* send_buffer_ptr = static_cast<const std::uint8_t*>(send_buffer->data());

    if (!needs_copy) {
      send_buffer_ptr +=
        (send_subdomain.lo() - source_->domain().lo()).dot(source_strides) * field_size_;
    }
    LEGATE_CHECK_NCCL(
      ncclSend(send_buffer_ptr, send_counts[idx], nccl_data_type, idx, comm, stream));
  }

  for (auto&& [idx, recv_subdomain] : enumerate(recv_subdomains)) {
    if (recv_counts[idx] == 0) {
      continue;
    }

    auto&& [recv_buffer, needs_copy] = recv_buffers[idx];

    auto* recv_buffer_ptr = static_cast<std::uint8_t*>(recv_buffer->data());

    if (!needs_copy) {
      recv_buffer_ptr +=
        (recv_subdomain.lo() - target_->domain().lo()).dot(target_strides) * field_size_;
    }
    LEGATE_CHECK_NCCL(
      ncclRecv(recv_buffer_ptr, recv_counts[idx], nccl_data_type, idx, comm, stream));
  }

  LEGATE_CHECK_NCCL(ncclGroupEnd());

  copy_from_recv_buffer_gpu_(field_size_,
                             *target_,
                             target_domain,
                             target_strides,
                             recv_buffers,
                             recv_subdomains,
                             recv_counts,
                             stream);
}

}  // namespace legate::detail
