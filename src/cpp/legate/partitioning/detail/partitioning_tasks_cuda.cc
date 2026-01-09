/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/cuda/detail/cuda_util.h>
#include <legate/generated/fatbin/partitioning_tasks_fatbin.h>
#include <legate/partitioning/detail/partitioning_tasks.h>
#include <legate/runtime/detail/runtime.h>
#include <legate/task/task_context.h>
#include <legate/utilities/detail/cuda_reduction_buffer.cuh>
#include <legate/utilities/detail/unravel.h>

#include <fmt/format.h>

#include <cstring>

namespace legate::detail {

// clang++ 18.x throws a warning that it cannot see the instantiation of template
// class ElementWiseMin/Max
// So, adding extern refs to such explicit instantiations
//
// Relevant compiler error log:
// src/cpp/legate/partitioning/detail/partitioning_tasks_cuda.cc:82:51: error:
// instantiation of variable 'legate::detail::ElementWiseMin<1>::identity' required
// here, but no definition is available [-Werror,-Wundefined-var-template]
//    82 |       auto ident_lo = ElementWiseMin<POINT_NDIM>::identity;
// src/cpp/legate/partitioning/detail/partitioning_tasks_cuda.cc:82:51: note: add
// an explicit instantiation declaration to suppress this warning if
// 'legate::detail::ElementWiseMin<1>::identity' is explicitly instantiated in
// another translation unit
//    82 |       auto ident_lo = ElementWiseMin<POINT_NDIM>::identity;
//
#define LEGATE_ADD_ELEMENTWISE_OP_INST_REF(N)                 \
  extern template const Point<N> ElementWiseMin<N>::identity; \
  extern template const Point<N> ElementWiseMax<N>::identity;

LEGION_FOREACH_N(LEGATE_ADD_ELEMENTWISE_OP_INST_REF)

#undef LEGATE_ADD_ELEMENTWISE_OP_INST_REF

namespace {

[[nodiscard]] constexpr std::size_t round_div(std::size_t x, std::size_t d)
{
  return (x + d - 1) / d;
}

template <bool RECT>
struct FindBoundingBoxFn {
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::TaskContext& context,
                  const legate::PhysicalStore& input,
                  const legate::PhysicalStore& output)
  {
    auto shape   = input.shape<STORE_NDIM>();
    auto out_acc = output.write_accessor<Domain, 1>();

    auto stream = context.get_task_stream();

    auto unravel = Unravel<STORE_NDIM>{shape};

    auto result_low  = CUDAReductionBuffer<ElementWiseMin<POINT_NDIM>>{stream};
    auto result_high = CUDAReductionBuffer<ElementWiseMax<POINT_NDIM>>{stream};

    const std::size_t blocks = round_div(unravel.volume(), LEGATE_THREADS_PER_BLOCK);
    const std::size_t shmem_size =
      LEGATE_THREADS_PER_BLOCK / LEGATE_WARP_SIZE * sizeof(Point<POINT_NDIM>) * 2;
    std::size_t num_iters = round_div(blocks, LEGATE_MAX_REDUCTION_CTAS);

    auto&& api         = cuda::detail::get_cuda_driver_api();
    auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();

    if (!unravel.empty()) {
      static const auto kernel_name = fmt::format("legate_find_bounding_box_kernel_{}_{}_{}",
                                                  RECT ? "rect" : "point",
                                                  POINT_NDIM,
                                                  STORE_NDIM);
      CUkernel kern =
        mod_manager.load_kernel_from_fatbin(partitioning_tasks_fatbin, kernel_name.c_str());

      auto ident_lo = ElementWiseMin<POINT_NDIM>::identity;
      auto ident_hi = ElementWiseMax<POINT_NDIM>::identity;

      if constexpr (RECT) {
        auto in_acc = input.read_accessor<Rect<POINT_NDIM>, STORE_NDIM>(shape);

        api->launch_kernel(kern,
                           {LEGATE_MAX_REDUCTION_CTAS},
                           {LEGATE_THREADS_PER_BLOCK},
                           shmem_size,
                           stream,
                           unravel,
                           num_iters,
                           result_low,
                           result_high,
                           in_acc,
                           ident_lo,
                           ident_hi);
      } else {
        auto in_acc = input.read_accessor<Point<POINT_NDIM>, STORE_NDIM>(shape);

        api->launch_kernel(kern,
                           {LEGATE_MAX_REDUCTION_CTAS},
                           {LEGATE_THREADS_PER_BLOCK},
                           shmem_size,
                           stream,
                           unravel,
                           num_iters,
                           result_low,
                           result_high,
                           in_acc,
                           ident_lo,
                           ident_hi);
      }
    }

    static const auto kernel_name = fmt::format("legate_copy_output_{}", POINT_NDIM);
    CUkernel kern =
      mod_manager.load_kernel_from_fatbin(partitioning_tasks_fatbin, kernel_name.c_str());

    api->launch_kernel(kern, {1}, {1}, 0, stream, out_acc, result_low, result_high);
  }
};

template <bool RECT>
struct FindBoundingBoxSortedFn {
  template <std::int32_t POINT_NDIM, std::int32_t STORE_NDIM>
  void operator()(const legate::TaskContext& context,
                  const legate::PhysicalStore& input,
                  const legate::PhysicalStore& output)
  {
    auto shape   = input.shape<STORE_NDIM>();
    auto out_acc = output.write_accessor<Domain, 1>();

    auto stream = context.get_task_stream();

    auto unravel = Unravel<STORE_NDIM>{shape};

    auto result_low  = CUDAReductionBuffer<ElementWiseMin<POINT_NDIM>>{stream};
    auto result_high = CUDAReductionBuffer<ElementWiseMax<POINT_NDIM>>{stream};

    auto&& api         = cuda::detail::get_cuda_driver_api();
    auto&& mod_manager = Runtime::get_runtime().get_cuda_module_manager();

    if (!unravel.empty()) {
      static const auto kernel_name = fmt::format("legate_find_bounding_box_sorted_kernel_{}_{}_{}",
                                                  RECT ? "rect" : "point",
                                                  POINT_NDIM,
                                                  STORE_NDIM);
      CUkernel kern =
        mod_manager.load_kernel_from_fatbin(partitioning_tasks_fatbin, kernel_name.c_str());

      if constexpr (RECT) {
        auto in_acc = input.read_accessor<Rect<POINT_NDIM>, STORE_NDIM>(shape);

        api->launch_kernel(kern, {1}, {1}, 0, stream, unravel, result_low, result_high, in_acc);
      } else {
        auto in_acc = input.read_accessor<Point<POINT_NDIM>, STORE_NDIM>(shape);

        api->launch_kernel(kern, {1}, {1}, 0, stream, unravel, result_low, result_high, in_acc);
      }
    }

    static const auto kernel_name = fmt::format("legate_copy_output_{}", POINT_NDIM);
    CUkernel kern =
      mod_manager.load_kernel_from_fatbin(partitioning_tasks_fatbin, kernel_name.c_str());

    api->launch_kernel(kern, {1}, {1}, 0, stream, out_acc, result_low, result_high);
  }
};

}  // namespace

/*static*/ void FindBoundingBox::gpu_variant(legate::TaskContext context)
{
  auto input  = context.input(0).data();
  auto output = context.output(0).data();

  auto type = input.type();

  if (legate::is_rect_type(type)) {
    legate::double_dispatch(
      legate::ndim_rect_type(type), input.dim(), FindBoundingBoxFn<true>{}, context, input, output);
  } else {
    legate::double_dispatch(legate::ndim_point_type(type),
                            input.dim(),
                            FindBoundingBoxFn<false>{},
                            context,
                            input,
                            output);
  }
}

/*static*/ void FindBoundingBoxSorted::gpu_variant(legate::TaskContext context)
{
  auto input  = context.input(0).data();
  auto output = context.output(0).data();

  auto type = input.type();

  if (legate::is_rect_type(type)) {
    legate::double_dispatch(legate::ndim_rect_type(type),
                            input.dim(),
                            FindBoundingBoxSortedFn<true>{},
                            context,
                            input,
                            output);
  } else {
    legate::double_dispatch(legate::ndim_point_type(type),
                            input.dim(),
                            FindBoundingBoxSortedFn<false>{},
                            context,
                            input,
                            output);
  }
}

}  // namespace legate::detail
