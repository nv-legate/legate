/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/data/detail/reduction_request.h>

#include <legate/comm/coll.h>
#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/data/detail/copy_util.h>
#include <legate/data/detail/inline_storage.h>
#include <legate/redop/base.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/type/type_traits.h>
#include <legate/type/types.h>
#include <legate/utilities/detail/geometry.h>
#include <legate/utilities/detail/traced_exception.h>
#include <legate/utilities/dispatch.h>

#include <algorithm>
#include <cstddef>
#include <nccl.h>
#include <stdexcept>
#include <type_traits>

namespace legate::detail {

namespace {

template <typename T>
void fill_typed(T* typed, std::size_t count, ReductionOpKind op_kind)
{
  switch (op_kind) {
    case ReductionOpKind::ADD: {
      std::fill_n(typed, count, legate::SumReduction<T>::identity);
      return;
    }
    case ReductionOpKind::MUL: {
      std::fill_n(typed, count, legate::ProdReduction<T>::identity);
      return;
    }
    case ReductionOpKind::MAX: {
      std::fill_n(typed, count, legate::MaxReduction<T>::identity);
      return;
    }
    case ReductionOpKind::MIN: {
      std::fill_n(typed, count, legate::MinReduction<T>::identity);
      return;
    }
    case ReductionOpKind::OR: {
      if constexpr (std::is_integral_v<T>) {
        std::fill_n(typed, count, legate::OrReduction<T>::identity);
        return;
      }
      throw TracedException<std::invalid_argument>{
        "Reduction identity does not support non-integral types with OR"};
    }
    case ReductionOpKind::AND: {
      if constexpr (std::is_integral_v<T>) {
        std::fill_n(typed, count, legate::AndReduction<T>::identity);
        return;
      }
      throw TracedException<std::invalid_argument>{
        "Reduction identity does not support non-integral types with AND"};
    }
    case ReductionOpKind::XOR: {
      if constexpr (std::is_integral_v<T>) {
        std::fill_n(typed, count, legate::XORReduction<T>::identity);
        return;
      }
      throw TracedException<std::invalid_argument>{
        "Reduction identity does not support non-integral types with XOR"};
    }
  }
  LEGATE_UNREACHABLE();
}

class FillTypedFn {
 public:
  template <Type::Code CODE>
  void operator()(void* dst, std::size_t count, ReductionOpKind op_kind) const
  {
    if constexpr (CODE == Type::Code::INT8 || CODE == Type::Code::INT32 ||
                  CODE == Type::Code::INT64 || CODE == Type::Code::UINT8 ||
                  CODE == Type::Code::UINT32 || CODE == Type::Code::UINT64 ||
                  CODE == Type::Code::FLOAT32 || CODE == Type::Code::FLOAT64) {
      using T = type_of_t<CODE>;
      fill_typed<T>(static_cast<T*>(dst), count, op_kind);
    } else {
      // The set of supported type codes mirrors `to_coll_data_type_()` in
      // copy_util.cc. Keep these in sync.
      throw TracedException<std::runtime_error>{"Not supported yet"};
    }
  }
};

struct CopyTileIntoFullStoreCPUFn {
  template <Type::Code CODE>
  void operator()(InlineStorage& full, const Domain& full_domain, const InlineStorage& tile) const
  {
    if constexpr (CODE == Type::Code::INT8 || CODE == Type::Code::INT32 ||
                  CODE == Type::Code::INT64 || CODE == Type::Code::UINT8 ||
                  CODE == Type::Code::UINT32 || CODE == Type::Code::UINT64 ||
                  CODE == Type::Code::FLOAT32 || CODE == Type::Code::FLOAT64) {
      using T = type_of_t<CODE>;

      const auto& tile_domain = tile.domain();
      if (tile_domain.get_volume() == 0) {
        return;
      }
      const auto full_strides = calculate_strides(full_domain);
      const auto tile_strides = calculate_strides(tile_domain);

      auto* full_ptr = static_cast<T*>(full.data()) - inner_product(full_domain.lo(), full_strides);
      const auto* tile_ptr =
        static_cast<const T*>(tile.data()) - inner_product(tile_domain.lo(), tile_strides);

      for (Domain::DomainPointIterator it{tile_domain}; it; ++it) {
        full_ptr[inner_product(it.p, full_strides)] = tile_ptr[inner_product(it.p, tile_strides)];
      }
    } else {
      throw TracedException<std::runtime_error>{"Not supported yet"};
    }
  }
};

}  // namespace

void fill_with_reduction_identity(void* dst,
                                  std::size_t count,
                                  Type::Code type_code,
                                  ReductionOpKind op_kind)
{
  if (count == 0) {
    return;
  }
  type_dispatch(type_code, FillTypedFn{}, dst, count, op_kind);
}

void ReductionRequest::perform(VariantCode variant_code) const
{
  // The GPU path currently funnels every scenario through `all_reduce_` (ncclAllReduce).
  // Partition-aware GPU paths will be added when the GPU reduction rework lands.
  if (variant_code == VariantCode::GPU) {
    all_reduce_(variant_code);
    return;
  }
  LEGATE_ASSERT(variant_code == VariantCode::CPU);

  const auto target_partitioned       = target_partition_->has_color_shape();
  const auto source_partitioned       = source_partition_->has_color_shape();
  const auto source_covers_full_store = source_->domain().get_volume() == sub_domain_.get_volume();

  if (!target_partitioned && source_covers_full_store) {
    all_reduce_(variant_code);
    return;
  }

  if (!source_partitioned) {
    LEGATE_ASSERT(target_partitioned);
    throw TracedException<std::runtime_error>{
      "Reduction with NoPartition source into a Tiling target is not yet supported"};
  }

  if (!target_partitioned) {
    reduce_to_replicated_target_(variant_code);
    return;
  }

  throw TracedException<std::runtime_error>{
    "Reduction with Tiling source into a misaligned Tiling target is not yet supported"};
}

void ReductionRequest::all_reduce_(VariantCode variant_code) const
{
  if (variant_code == VariantCode::GPU) {
    const cuda::detail::AutoPrimaryContext ctx{Runtime::get_runtime().get_current_cuda_device()};

    // TODO(wonchanl): Error checking
    ncclAllReduce(source_->data(),
                  target_->data(),
                  source_->domain().get_volume(),
                  to_nccl_data_type(type_->code),
                  to_nccl_red_op(op_kind_),
                  Runtime::get_runtime().get_gpu_comm(),
                  Runtime::get_runtime().get_cuda_stream());
    return;
  }

  LEGATE_ASSERT(variant_code == VariantCode::CPU);

  collAllreduce(source_->data(),
                target_->data(),
                static_cast<int>(source_->domain().get_volume()),
                to_coll_data_type_(type_->code),
                op_kind_,
                Runtime::get_runtime().get_cpu_comm());
}

void ReductionRequest::reduce_to_replicated_target_(VariantCode variant_code) const
{
  LEGATE_ASSERT(variant_code == VariantCode::CPU);
  static_cast<void>(variant_code);

  const auto full_volume = sub_domain_.get_volume();
  if (full_volume == 0) {
    return;
  }

  InlineStorage local_full{sub_domain_, type_->size(), source_->target()};
  fill_with_reduction_identity(local_full.data(), full_volume, type_->code, op_kind_);

  type_dispatch(type_->code, CopyTileIntoFullStoreCPUFn{}, local_full, sub_domain_, *source_);

  collAllreduce(local_full.data(),
                target_->data(),
                static_cast<int>(full_volume),
                to_coll_data_type_(type_->code),
                op_kind_,
                Runtime::get_runtime().get_cpu_comm());
}

}  // namespace legate::detail
