/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/reduction_request.h>

#include <legate/comm/coll.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/data/detail/buffer.h>
#include <legate/data/detail/copy_util.h>
#include <legate/generated/fatbin/reduction_request_fatbin.h>
#include <legate/partitioning/detail/partition/tiling.h>
#include <legate/redop/base.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/type_traits.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/dispatch.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <nccl.h>
#include <stdexcept>
#include <type_traits>

namespace legate::detail {

namespace {

template <ReductionOpKind REDOP, Type::Code CODE>
struct reduction_identity;

template <Type::Code CODE>
struct reduction_identity<ReductionOpKind::ADD, CODE> {
  using VAL                   = type_of_t<CODE>;
  static constexpr auto value = SumReduction<VAL>::identity;
};

template <Type::Code CODE>
struct reduction_identity<ReductionOpKind::MUL, CODE> {
  using VAL                   = type_of_t<CODE>;
  static constexpr auto value = ProdReduction<VAL>::identity;
};

template <Type::Code CODE>
struct reduction_identity<ReductionOpKind::MAX, CODE> {
  using VAL                   = type_of_t<CODE>;
  static constexpr auto value = MaxReduction<VAL>::identity;
};

template <Type::Code CODE>
struct reduction_identity<ReductionOpKind::MIN, CODE> {
  using VAL                   = type_of_t<CODE>;
  static constexpr auto value = MinReduction<VAL>::identity;
};

template <Type::Code CODE>
struct reduction_identity<ReductionOpKind::OR, CODE> {
  using VAL                   = type_of_t<CODE>;
  static constexpr auto value = OrReduction<VAL>::identity;
};

template <Type::Code CODE>
struct reduction_identity<ReductionOpKind::AND, CODE> {
  using VAL                   = type_of_t<CODE>;
  static constexpr auto value = AndReduction<VAL>::identity;
};

template <Type::Code CODE>
struct reduction_identity<ReductionOpKind::XOR, CODE> {
  using VAL                   = type_of_t<CODE>;
  static constexpr auto value = XORReduction<VAL>::identity;
};

template <ReductionOpKind REDOP, Type::Code CODE>
inline constexpr decltype(auto) reduction_identity_v = reduction_identity<REDOP, CODE>::value;

constexpr bool supports_reduction(Type::Code code)
{
  // The set of supported type codes mirrors `to_coll_data_type_()` in
  // copy_util.cc. Keep these in sync.
  return code == Type::Code::INT8 || code == Type::Code::INT32 || code == Type::Code::INT64 ||
         code == Type::Code::UINT8 || code == Type::Code::UINT32 || code == Type::Code::UINT64 ||
         code == Type::Code::FLOAT32 || code == Type::Code::FLOAT64;
}

template <ReductionOpKind REDOP, Type::Code CODE>
struct valid_reduction;

template <Type::Code CODE>
struct valid_reduction<ReductionOpKind::ADD, CODE> {
  static constexpr bool value = supports_reduction(CODE);
};

template <Type::Code CODE>
struct valid_reduction<ReductionOpKind::MUL, CODE> {
  static constexpr bool value = supports_reduction(CODE);
};

template <Type::Code CODE>
struct valid_reduction<ReductionOpKind::MAX, CODE> {
  static constexpr bool value = supports_reduction(CODE);
};

template <Type::Code CODE>
struct valid_reduction<ReductionOpKind::MIN, CODE> {
  static constexpr bool value = supports_reduction(CODE);
};

template <Type::Code CODE>
struct valid_reduction<ReductionOpKind::OR, CODE> {
  static constexpr bool value = supports_reduction(CODE) && std::is_integral_v<type_of_t<CODE>>;
};

template <Type::Code CODE>
struct valid_reduction<ReductionOpKind::AND, CODE> {
  static constexpr bool value = supports_reduction(CODE) && std::is_integral_v<type_of_t<CODE>>;
};

template <Type::Code CODE>
struct valid_reduction<ReductionOpKind::XOR, CODE> {
  static constexpr bool value = supports_reduction(CODE) && std::is_integral_v<type_of_t<CODE>>;
};

template <ReductionOpKind REDOP, Type::Code CODE>
inline constexpr bool valid_reduction_v = valid_reduction<REDOP, CODE>::value;

template <typename Functor, typename... Fnargs>
constexpr decltype(auto) redop_dispatch(ReductionOpKind op_kind, Functor f, Fnargs&&... args)
{
  switch (op_kind) {
    case ReductionOpKind::ADD: {
      return f.template operator()<ReductionOpKind::ADD>(std::forward<Fnargs>(args)...);
    }
    case ReductionOpKind::MUL: {
      return f.template operator()<ReductionOpKind::MUL>(std::forward<Fnargs>(args)...);
    }
    case ReductionOpKind::MAX: {
      return f.template operator()<ReductionOpKind::MAX>(std::forward<Fnargs>(args)...);
    }
    case ReductionOpKind::MIN: {
      return f.template operator()<ReductionOpKind::MIN>(std::forward<Fnargs>(args)...);
    }
    case ReductionOpKind::OR: {
      return f.template operator()<ReductionOpKind::OR>(std::forward<Fnargs>(args)...);
    }
    case ReductionOpKind::AND: {
      return f.template operator()<ReductionOpKind::AND>(std::forward<Fnargs>(args)...);
    }
    case ReductionOpKind::XOR: {
      return f.template operator()<ReductionOpKind::XOR>(std::forward<Fnargs>(args)...);
    }
  }
  LEGATE_UNREACHABLE();
  return f.template operator()<ReductionOpKind::ADD>(std::forward<Fnargs>(args)...);
}

template <Type::Code CODE>
class FillCPU {
 public:
  template <ReductionOpKind REDOP, std::enable_if_t<valid_reduction_v<REDOP, CODE>>* = nullptr>
  void operator()(void* dst, std::size_t count) const
  {
    using VAL = type_of_t<CODE>;

    auto* typed = static_cast<VAL*>(dst);
    std::fill_n(typed, count, reduction_identity_v<REDOP, CODE>);
  }

  template <ReductionOpKind REDOP, std::enable_if_t<!valid_reduction_v<REDOP, CODE>>* = nullptr>
  void operator()(void*, std::size_t) const
  {
    throw TracedException<std::invalid_argument>{"Unsupported reduction"};
  }
};

template <Type::Code CODE>
class FillGPU {
 public:
  template <ReductionOpKind REDOP, std::enable_if_t<valid_reduction_v<REDOP, CODE>>* = nullptr>
  void operator()(void* dst, std::size_t count) const
  {
    using VAL = type_of_t<CODE>;
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

    auto&& api = cuda::detail::get_cuda_driver_api();

    if constexpr (sizeof(VAL) == 1) {
      std::uint8_t value{};

      std::memcpy(&value, &reduction_identity_v<REDOP, CODE>, sizeof(VAL));
      api->mem_set_d8_async(dst, value, count, Runtime::get_runtime().get_cuda_stream());
    } else if constexpr (sizeof(VAL) == 4) {
      std::uint32_t value{};

      std::memcpy(&value, &reduction_identity_v<REDOP, CODE>, sizeof(VAL));
      api->mem_set_d32_async(dst, value, count, Runtime::get_runtime().get_cuda_stream());
    } else if constexpr (sizeof(VAL) == 8) {
      auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();
      auto kern =
        mod_manager.load_kernel_from_fatbin(reduction_request_fatbin, "legate_fill_kernel_64");
      const auto num_blocks = (count + LEGATE_THREADS_PER_BLOCK - 1) / LEGATE_THREADS_PER_BLOCK;
      std::uint64_t value{};

      std::memcpy(&value, &reduction_identity_v<REDOP, CODE>, sizeof(VAL));
      api->launch_kernel(kern,
                         {num_blocks},
                         {LEGATE_THREADS_PER_BLOCK},
                         /*shared_mem_bytes=*/0,
                         Runtime::get_runtime().get_cuda_stream(),
                         static_cast<std::uint64_t*>(dst),
                         count,
                         value);

    } else {
      throw TracedException<std::runtime_error>{"Unsupported type"};
    }
  }

  template <ReductionOpKind REDOP, std::enable_if_t<!valid_reduction_v<REDOP, CODE>>* = nullptr>
  void operator()(void*, std::size_t) const
  {
    throw TracedException<std::invalid_argument>{"Unsupported reduction"};
  }
};

class FillDispatch {
 public:
  template <Type::Code CODE>
  void operator()(void* dst,
                  std::size_t count,
                  mapping::StoreTarget target,
                  ReductionOpKind op_kind) const
  {
    if (target == mapping::StoreTarget::SYSMEM) {
      redop_dispatch(op_kind, FillCPU<CODE>{}, dst, count);
    } else {
      redop_dispatch(op_kind, FillGPU<CODE>{}, dst, count);
    }
  }
};

}  // namespace

void fill_with_reduction_identity(void* dst,
                                  std::size_t count,
                                  mapping::StoreTarget target,
                                  Type::Code type_code,
                                  ReductionOpKind op_kind)
{
  if (count == 0) {
    return;
  }

  type_dispatch(type_code, FillDispatch{}, dst, count, target, op_kind);
}

void ReductionRequest::perform(VariantCode variant_code) const
{
  const auto target_partitioned       = target_partition_->has_color_shape();
  const auto source_partitioned       = source_partition_->has_color_shape();
  const auto source_covers_full_store = source_->domain().volume() == sub_domain_.volume();

  // Replicated target: every rank holds a full-size partial that must be combined across ranks.
  if (!target_partitioned && source_covers_full_store) {
    all_reduce_(variant_code);
    return;
  }

  // Disjoint tiled target whose tiling matches the reduction's launch tiling: each rank's buffer
  // already holds the fully-reduced values for the (disjoint) tile it owns, so it just copies that
  // tile back into the canonical storage. No cross-rank communication is required.
  if (target_partitioned && source_covers_full_store) {
    const auto* source_tiling = dynamic_cast<const Tiling*>(source_partition_.get());
    const auto* target_tiling = dynamic_cast<const Tiling*>(target_partition_.get());

    if (source_tiling != nullptr && target_tiling != nullptr && *source_tiling == *target_tiling) {
      reduce_to_tiled_target_(variant_code);
      return;
    }
  }

  if (!source_partitioned) {
    LEGATE_ASSERT(target_partitioned);
    throw TracedException<std::runtime_error>{
      "Reduction with NoPartition source into a Tiling target is not yet supported"};
  }

  throw TracedException<std::runtime_error>{
    "Reduction with Tiling source into a misaligned Tiling target is not yet supported"};
}

void ReductionRequest::reduce_to_tiled_target_(VariantCode variant_code) const
{
  const auto target_domain = target_->domain();
  const auto count         = target_domain.volume();

  if (count == 0) {
    return;
  }

  const auto source_domain  = source_->domain();
  const auto source_strides = source_domain.strides();
  const auto source_extents = source_domain.extents();
  const auto target_extents = target_domain.extents();

  // The target tile occupies a contiguous run of the row-major source only when it spans the full
  // extent of every dimension but the outermost (true for the 1-D reduction stores handled here,
  // and for tilings along the outermost dimension). Refuse the strided case loudly rather than
  // silently corrupting data.
  for (std::uint32_t dim = 1; dim < source_domain.dim(); ++dim) {
    if (target_extents[dim] != source_extents[dim]) {
      throw TracedException<std::runtime_error>{
        "Reduction into a non-contiguous tile of the target is not yet supported"};
    }
  }

  // This rank's tile lives at its global coordinates within the full-size source buffer.
  const auto field_size    = type_->size();
  const auto source_offset = (target_domain.lo() - source_domain.lo()).dot(source_strides);
  const auto num_bytes     = count * field_size;
  const auto* source_ptr =
    static_cast<const std::int8_t*>(source_->data()) + (source_offset * field_size);
  auto* target_ptr = static_cast<std::int8_t*>(target_->data());

  if (variant_code == VariantCode::GPU) {
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

    cuda::detail::get_cuda_driver_api()->mem_cpy_async(
      target_ptr, source_ptr, num_bytes, Runtime::get_runtime().get_cuda_stream());
    return;
  }

  LEGATE_ASSERT(variant_code == VariantCode::CPU);
  std::memcpy(target_ptr, source_ptr, num_bytes);
}

void ReductionRequest::all_reduce_(VariantCode variant_code) const
{
  if (variant_code == VariantCode::GPU) {
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

    // TODO(wonchanl): Error checking
    ncclAllReduce(source_->data(),
                  target_->data(),
                  std::max<>(source_->domain().volume(), std::size_t{1}),
                  to_nccl_data_type(type_->code),
                  to_nccl_red_op(op_kind_),
                  Runtime::get_runtime().get_gpu_comm(),
                  Runtime::get_runtime().get_cuda_stream());
    return;
  }

  LEGATE_ASSERT(variant_code == VariantCode::CPU);

  collAllreduce(source_->data(),
                target_->data(),
                std::max<>(static_cast<int>(source_->domain().volume()), 1),
                to_coll_data_type_(type_->code),
                op_kind_,
                Runtime::get_runtime().get_cpu_comm());
}

}  // namespace legate::detail
