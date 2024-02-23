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

#pragma once

#include "core/utilities/assert.h"
#include "core/utilities/defined.h"

#include "legate.h"

#if LegateDefined(LEGATE_USE_CUDA) && LegateDefined(REALM_COMPILER_IS_NVCC)
#include "core/cuda/stream_pool.h"
#endif

#include "functional.hpp"
#include "meta.hpp"
#include "registrar.hpp"
#include "slice.hpp"
#include "store.hpp"

#include <atomic>
#include <vector>

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

namespace detail {

using logical_stores_t = const std::vector<LogicalStore>&;

template <typename... Ts>
struct inputs {
  using Types = meta::list<Ts...>;
  std::vector<LogicalStore> data;

  template <typename Type, typename Kind>
  void apply(AutoTask& task, Kind kind, std::size_t index) const
  {
    // Add the stores as inputs to the task
    Variable part = task.find_or_declare_partition(data[index]);
    task.add_input(data[index], part);

    // Add the partitioning constraints
    auto constraints = Type::policy::partition_constraints(kind);
    std::apply([&](auto... cs) { (task.add_constraint(cs(part)), ...); }, constraints);
  }

  template <typename Kind>
  void operator()(AutoTask& task, Kind kind) const
  {
    LegateAssert(data.size() == sizeof...(Ts));

    std::size_t index = 0;
    (this->apply<Ts>(task, kind, index++), ...);
  }
};

template <typename... Ts>
struct outputs {
  using Types = meta::list<Ts...>;
  std::vector<LogicalStore> data;

  template <typename Type, typename Kind>
  void apply(AutoTask& task, Kind kind, std::size_t index) const
  {
    // Add the stores as outputs to the task
    Variable part = task.find_or_declare_partition(data[index]);
    task.add_output(data[index], part);

    // Add the partitioning constraints
    auto constraints = Type::policy::partition_constraints(kind);
    std::apply([&](auto... cs) { (task.add_constraint(cs(part)), ...); }, constraints);
  }

  template <typename Kind>
  void operator()(AutoTask& task, Kind kind) const
  {
    static_cast<void>(kind);
    LegateAssert(data.size() == sizeof...(Ts));

    std::size_t index = 0;
    (this->apply<Ts>(task, kind, index++), ...);
  }
};

template <typename... Ts>
struct constraints {
  std::tuple<Ts...> data;

  void operator()(AutoTask& task,
                  logical_stores_t inputs,
                  logical_stores_t outputs,
                  const LogicalStore& reduction) const
  {
    std::apply([&](auto&&... cons) { (cons(task, inputs, outputs, reduction), ...); }, data);
  }
};

template <typename... Ts>
struct scalars {
  std::tuple<Ts...> data;

  void operator()(AutoTask& task) const
  {
    std::apply(
      [&](auto&... scalar) {
        (task.add_scalar_arg(Scalar(binary_type(sizeof(scalar)), &scalar, true)), ...);
      },
      data);
  }
};

template <typename... Fn>
struct function;

template <>
struct function<> {};

template <typename Fn>
struct function<Fn> {
  Fn fn;

  void operator()(AutoTask& task) const
  {
    task.add_scalar_arg(Scalar(binary_type(sizeof(fn)), &fn, true));
  }
};

inline std::int32_t _next_reduction_id()
{
  static std::atomic<std::int32_t> id{LEGION_REDOP_LAST};
  return id.fetch_add(1);
}

template <typename T>
std::int32_t _reduction_id_for()
{
  static const std::int32_t id = _next_reduction_id();
  return id;
}

inline std::int32_t _next_reduction_kind()
{
  static std::atomic<std::int32_t> id{static_cast<std::int32_t>(ReductionOpKind::XOR) + 1};
  return id.fetch_add(1);
}

template <typename T>
std::int32_t _reduction_kind_for()
{
  static const std::int32_t id = _next_reduction_kind();
  return id;
}

template <typename Fun>
std::int32_t _get_reduction_id()
{
  static const std::int32_t ID = []() -> std::int32_t {
    std::int32_t id               = _reduction_id_for<Fun>();
    observer_ptr<Runtime> runtime = Runtime::get_runtime();
    Library library = runtime->find_or_create_library("legate.stl", LEGATE_STL_RESOURCE_CONFIG);

    return library.register_reduction_operator<Fun>(id);
  }();
  return ID;
}

template <typename ElementType, typename Fun>
std::int32_t _record_reduction_for()
{
  static const std::int32_t KIND = []() -> std::int32_t {
    Type type(primitive_type(type_code_of<ElementType>));
    std::int32_t id   = _get_reduction_id<Fun>();
    std::int32_t kind = _reduction_kind_for<Fun>();
    type.record_reduction_operator(kind, id);
    return kind;
  }();
  return KIND;
}

template <typename...>
struct reduction;

template <>
struct reduction<> {};

template <typename Store, typename Fun>
struct reduction<Store, Fun> {
  LogicalStore data;
  Fun fn;

  void operator()(AutoTask& task) const
  {
    auto part = task.find_or_declare_partition(data);

    std::int32_t kind = _record_reduction_for<element_type_of_t<Store>, Fun>();
    task.add_reduction(data, kind, part);

    task.add_scalar_arg(Scalar(binary_type(sizeof(fn)), &fn, true));
  }
};

enum class store_type : int { input, output, reduction };

struct store_placeholder {
  store_type which;
  int index;

  LogicalStore operator()(logical_stores_t inputs,
                          logical_stores_t outputs,
                          const LogicalStore& reduction) const
  {
    if (which == store_type::input) {
      return inputs[index];
    } else if (which == store_type::output) {
      return outputs[index];
    } else {
      return reduction;
    }
  }
};

struct make_inputs {
  template <typename... Ts>                  //
    requires(logical_store_like<Ts> && ...)  //
  inputs<std::remove_reference_t<Ts>...> operator()(Ts&&... stores) const
  {
    return {std::vector<LogicalStore>{get_logical_store(std::forward<Ts>(stores))...}};
  }

  store_placeholder operator[](int index) const { return {store_type::input, index}; }

  logical_stores_t operator()(logical_stores_t inputs,
                              logical_stores_t /*outputs*/,
                              const LogicalStore& /*reduction*/) const
  {
    return inputs;
  }
};

struct make_outputs {
  template <typename... Ts>                  //
    requires(logical_store_like<Ts> && ...)  //
  outputs<std::remove_reference_t<Ts>...> operator()(Ts&&... stores) const
  {
    return {std::vector<LogicalStore>{get_logical_store(std::forward<Ts>(stores))...}};
  }

  store_placeholder operator[](int index) const { return {store_type::output, index}; }

  logical_stores_t operator()(logical_stores_t /*inputs*/,
                              logical_stores_t outputs,
                              const LogicalStore& /*reduction*/) const
  {
    return outputs;
  }
};

struct make_scalars {
  template <typename... Ts>
  scalars<Ts...> operator()(Ts... scalars) const
  {
    static_assert((std::is_trivially_copyable_v<Ts> && ...),
                  "All scalar arguments must be trivially copyable");
    return {{static_cast<Ts&&>(scalars)...}};
  }
};

struct make_function {
  template <typename Fn>
  function<Fn> operator()(Fn fn) const
  {
    return {static_cast<Fn&&>(fn)};
  }
};

template <typename Store>
constexpr std::int32_t _dim_of() noexcept
{
  return dim_of_v<Store>;
}

template <typename Store>
using dim_of_t = meta::constant<_dim_of<Store>()>;

struct make_reduction : store_placeholder {
  constexpr make_reduction()  //
    : store_placeholder{store_type::reduction, 0}
  {
  }

  using store_placeholder::operator();

  template <typename Store, typename ReductionFn>  //
    requires(logical_store_like<Store>)            // TODO constrain Fun
  reduction<std::remove_reference_t<Store>, ReductionFn> operator()(Store&& store,
                                                                    ReductionFn reduction) const
  {
    static_assert(legate_reduction<ReductionFn>,
                  "The stl::reduction() function requires a Legate reduction operation "
                  "such as legate::SumReduction or legate::MaxReduction");
    return {get_logical_store(std::forward<Store>(store)), reduction};
  }
};

struct make_constraints {
  template <typename... Ts>  //
    requires((callable<Ts, AutoTask&, logical_stores_t, logical_stores_t, const LogicalStore&> &&
              ...))
  constraints<Ts...> operator()(Ts... constraints) const
  {
    return {{static_cast<Ts&&>(constraints)...}};
  }
};

struct make_align {
  static void do_align(AutoTask& task, Variable left, Variable right)
  {
    if (left.impl() != right.impl()) {
      task.add_constraint(legate::align(left, right));
    }
  }

  static void do_align(AutoTask& task, LogicalStore left, LogicalStore right)
  {
    do_align(task, task.find_or_declare_partition(left), task.find_or_declare_partition(right));
  }

  static void do_align(AutoTask& task, LogicalStore left, logical_stores_t right)
  {
    auto left_part = task.find_or_declare_partition(left);
    for (auto& store : right) {
      do_align(task, left_part, task.find_or_declare_partition(store));
    }
  }

  static void do_align(AutoTask& task, logical_stores_t left, LogicalStore right)
  {
    auto right_part = task.find_or_declare_partition(right);
    for (auto& store : left) {
      do_align(task, task.find_or_declare_partition(store), right_part);
    }
  }

  template <typename Left, typename Right>
  struct align {
    Left left;
    Right right;

    void operator()(AutoTask& task,
                    logical_stores_t inputs,
                    logical_stores_t outputs,
                    const LogicalStore& reduction) const
    {
      do_align(task, left(inputs, outputs, reduction), right(inputs, outputs, reduction));
    }
  };

  // E.g., `align(inputs[0], inputs[1])`
  //       `align(outputs[0], inputs)`
  template <typename Left, typename Right>
    requires(callable<Left, logical_stores_t, logical_stores_t, const LogicalStore&> &&
             callable<Right, logical_stores_t, logical_stores_t, const LogicalStore&>)  //
  align<Left, Right> operator()(Left left, Right right) const
  {
    return {left, right};
  }

  // For `align(inputs)`
  auto operator()(make_inputs inputs) const { return (*this)(inputs[0], inputs); }

  // For `align(outputs)`
  auto operator()(make_outputs outputs) const { return (*this)(outputs[0], outputs); }
};

template <typename Fn, typename... Views>
void _cpu_for_each(Fn fn, Views... views)
{
  auto&& input0 = front_of(views...);
  auto distance = input0.end() - input0.begin();
  for (std::int64_t idx = 0; idx < distance; ++idx) {  //
    fn(*(views.begin() + idx)...);
  }
}

template <typename Function, typename inputs, typename Outputs, typename Scalars>
struct iteration_cpu;

// This is a CPU implementation of a for_each operation.
template <typename Fn, typename... Is, typename... Os, typename... Ss>
struct iteration_cpu<function<Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
  template <std::size_t... IIs, std::size_t... OIs, std::size_t... SIs>
  static void impl(std::index_sequence<IIs...>,
                   std::index_sequence<OIs...>,
                   std::index_sequence<SIs...>,
                   const std::vector<PhysicalArray>& inputs,
                   std::vector<PhysicalArray>& outputs,
                   const std::vector<Scalar>& scalars)
  {
    _cpu_for_each(stl::bind_back(scalar_cast<const Fn&>(scalars[0]),
                                 scalar_cast<const Ss&>(scalars[SIs + 1])...),
                  Is::policy::physical_view(
                    as_mdspan<const stl::value_type_of_t<Is>, stl::dim_of_v<Is>>(inputs[IIs]))...,
                  Os::policy::physical_view(
                    as_mdspan<stl::value_type_of_t<Os>, stl::dim_of_v<Os>>(outputs[OIs]))...);
  }

  template <std::int32_t ActualDim>
  void operator()(const std::vector<PhysicalArray>& inputs,
                  std::vector<PhysicalArray>& outputs,
                  const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t Dim = dim_of_v<meta::front<Is...>>;

    if constexpr (Dim == ActualDim) {
      const Legion::Rect<Dim> shape = inputs[0].shape<Dim>();

      if (!shape.empty()) {
        impl(std::index_sequence_for<Is...>(),
             std::index_sequence_for<Os...>(),
             std::index_sequence_for<Ss...>(),
             inputs,
             outputs,
             scalars);
      }
    }
  }
};

#if LegateDefined(LEGATE_USE_CUDA) && LegateDefined(REALM_COMPILER_IS_NVCC)

template <typename Fn, typename... Views>
__global__ void _gpu_for_each(Fn fn, Views... views)
{
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  auto&& input0 = front_of(views...);
  auto distance = input0.end() - input0.begin();
  if (idx < distance) {  //
    fn(*(views.begin() + idx)...);
  }
}

template <typename Function, typename inputs, typename Outputs, typename Scalars>
struct iteration_gpu;

// This is a GPU implementation of a for_each operation.
template <typename Fn, typename... Is, typename... Os, typename... Ss>
struct iteration_gpu<function<Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
  static constexpr std::int32_t THREAD_BLOCK_SIZE = 128;

  template <std::size_t... IIs, std::size_t... OIs, std::size_t... SIs>
  static void impl(std::index_sequence<IIs...>,
                   std::index_sequence<OIs...>,
                   std::index_sequence<SIs...>,
                   const std::vector<PhysicalArray>& inputs,
                   std::vector<PhysicalArray>& outputs,
                   const std::vector<Scalar>& scalars)
  {
    const cuda::StreamView stream = cuda::StreamPool::get_stream_pool().get_stream();
    const std::size_t volume      = meta::front<Is...>::policy::size(inputs[0]);
    const std::size_t num_blocks  = (volume + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    _gpu_for_each<<<num_blocks, THREAD_BLOCK_SIZE, 0, stream>>>(
      stl::bind_back(scalar_cast<const Fn&>(scalars[0]),
                     scalar_cast<const Ss&>(scalars[SIs + 1])...),
      Is::policy::physical_view(
        as_mdspan<const stl::value_type_of_t<Is>, stl::dim_of_v<Is>>(inputs[IIs]))...,
      Os::policy::physical_view(
        as_mdspan<stl::value_type_of_t<Os>, stl::dim_of_v<Os>>(outputs[OIs]))...);
  }

  template <std::int32_t ActualDim>
  void operator()(const std::vector<PhysicalArray>& inputs,
                  std::vector<PhysicalArray>& outputs,
                  const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t Dim = dim_of_v<meta::front<Is...>>;

    if constexpr (Dim == ActualDim) {
      const Legion::Rect<Dim> shape = inputs[0].shape<Dim>();

      if (!shape.empty()) {
        impl(std::index_sequence_for<Is...>(),
             std::index_sequence_for<Os...>(),
             std::index_sequence_for<Ss...>(),
             inputs,
             outputs,
             scalars);
      }
    }
  }
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// iteration operation wrapper implementation
template <typename Function,
          typename Inputs,
          typename Outputs,
          typename Scalars,
          typename Constraints>
struct iteration_operation  //
  : LegateTask<iteration_operation<Function, Inputs, Outputs, Scalars, Constraints>> {
  static void cpu_variant(TaskContext context)
  {
    const auto inputs  = context.inputs();
    auto outputs       = context.outputs();
    const auto scalars = context.scalars();

    dim_dispatch(inputs.at(0).dim(),
                 iteration_cpu<Function, Inputs, Outputs, Scalars>(),
                 inputs,
                 outputs,
                 scalars);
  }

#if LegateDefined(LEGATE_USE_CUDA) && LegateDefined(REALM_COMPILER_IS_NVCC)
  // FIXME(wonchanl): In case where this template is instantiated multiple times with the exact same
  // template arguments, the exact class definition changes depending on what compiler is compiling
  // this header, which could lead to inconsistent class definitions across compile units.
  // Unfortunately, the -Wall flag doesn't allow us to have a member declaration having no
  // definition, so we can't fix the problem simply by pre-declaring this member all the time. The
  // right fix for this is to allow task variants to be defined in separate classes, instead of
  // requiring them to be members of the same class.
  static void gpu_variant(TaskContext context);
  {
    const auto inputs  = context.inputs();
    auto outputs       = context.outputs();
    const auto scalars = context.scalars();

    dim_dispatch(inputs.at(0).dim(),
                 iteration_gpu<Function, Inputs, Outputs, Scalars>(),
                 inputs,
                 outputs,
                 scalars);
  }
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Op, std::int32_t Dim, bool Exclusive = false>
inline auto as_mdspan_reduction(PhysicalArray& array, Rect<Dim> working_set)
  -> mdspan_reduction_t<Op, Dim, Exclusive>
{
  PhysicalStore store = array.data();

  using Mapping = std::layout_right::mapping<std::dextents<coord_t, Dim>>;
  Mapping mapping{detail::dynamic_extents<Dim>(working_set)};

  using Policy   = reduction_accessor<Op, Exclusive>;
  using Accessor = detail::mdspan_accessor<typename Op::RHS, Dim, Policy>;
  Accessor accessor{store, working_set};

  using Handle = typename Accessor::data_handle_type;
  Handle handle{};

  return mdspan_reduction_t<Op, Dim, Exclusive>{handle, mapping, accessor};
}

template <typename Function, typename InOut, typename Input>
void _cpu_reduce(Function fn, InOut inout, Input input)
{
  auto distance = inout.end() - inout.begin();
  LegateAssert(distance == (input.end() - input.begin()));

  for (std::int64_t idx = 0; idx < distance; ++idx) {
    fn(*(inout.begin() + idx), *(input.begin() + idx));
  }
}

template <typename Reduction, typename Inputs, typename Outputs, typename Scalars>
struct reduction_cpu;

template <typename Red, typename Fn, typename... Is, typename... Os, typename... Ss>
struct reduction_cpu<reduction<Red, Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
  template <std::size_t... IIs, std::size_t... OIs, std::size_t... SIs>
  static void impl(std::index_sequence<IIs...>,
                   std::index_sequence<OIs...>,
                   std::index_sequence<SIs...>,
                   PhysicalArray& reduction,
                   const std::vector<PhysicalArray>& inputs,
                   std::vector<PhysicalArray>& outputs,
                   const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t Dim = stl::dim_of_v<Red>;
    Rect<Dim> working_set      = reduction.shape<Dim>();
    ((working_set = working_set.intersection(inputs[IIs].shape<Dim>())), ...);

    _cpu_reduce(stl::bind_back(scalar_cast<const Fn&>(scalars[0]),
                               scalar_cast<const Ss&>(scalars[SIs + 1])...),
                Red::policy::physical_view(  //
                  as_mdspan_reduction<Fn, Dim>(reduction, working_set)),
                Is::policy::physical_view(
                  as_mdspan<const stl::value_type_of_t<Is>, stl::dim_of_v<Is>>(inputs[IIs]))...,
                Os::policy::physical_view(
                  as_mdspan<stl::value_type_of_t<Os>, stl::dim_of_v<Os>>(outputs[OIs]))...);
  }

  template <std::int32_t ActualDim>
  void operator()(std::vector<PhysicalArray>& reductions,
                  const std::vector<PhysicalArray>& inputs,
                  std::vector<PhysicalArray>& outputs,
                  const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t Dim = dim_of_v<Red>;

    if constexpr (Dim == ActualDim) {
      const Legion::Rect<Dim> shape = reductions.at(0).shape<Dim>();

      if (!shape.empty()) {
        impl(std::index_sequence_for<Is...>(),
             std::index_sequence_for<Os...>(),
             std::index_sequence_for<Ss...>(),
             reductions.at(0),
             inputs,
             outputs,
             scalars);
      }
    }
  }
};

#if LegateDefined(LEGATE_USE_CUDA) && LegateDefined(REALM_COMPILER_IS_NVCC)

// TODO: this can be parallelized as well with care to avoid data races.
// If the view types carried metadata about the stride that avoids interference,
// then we can launch several kernels, each of which folds in parallel at
// multiples of that stride, but starting at different offsets. Then those
// results can be folded together.
template <typename Function, typename InOut, typename Input>
__global__ void _gpu_reduce(Function fn, InOut inout, Input input)
{
  const std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  const auto distance = inout.end() - inout.begin();
  LegateAssert(distance == (input.end() - input.begin()));

  for (std::int64_t idx = 0; idx < distance; ++idx) {
    fn(tid, *(inout.begin() + idx), *(input.begin() + idx));
  }
}

template <typename Reduction, typename Inputs, typename Outputs, typename Scalars>
struct reduction_gpu;

template <typename Red, typename Fn, typename... Is, typename... Os, typename... Ss>
struct reduction_gpu<reduction<Red, Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
  static constexpr std::int32_t THREAD_BLOCK_SIZE = 128;

  template <std::size_t... IIs, std::size_t... OIs, std::size_t... SIs>
  static void impl(std::index_sequence<IIs...>,
                   std::index_sequence<OIs...>,
                   std::index_sequence<SIs...>,
                   PhysicalArray& reduction,
                   const std::vector<PhysicalArray>& inputs,
                   std::vector<PhysicalArray>& outputs,
                   const std::vector<Scalar>& scalars)
  {
    const cuda::StreamView stream = cuda::StreamPool::get_stream_pool().get_stream();
    const std::size_t volume      = Red::policy::size(reduction);
    const std::size_t num_blocks  = (volume + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    constexpr std::int32_t Dim = dim_of_v<Red>;
    Rect<Dim> working_set      = reduction.shape<Dim>();
    ((working_set = working_set.intersection(inputs[IIs].shape<Dim>())), ...);

    _gpu_reduce<<<num_blocks, THREAD_BLOCK_SIZE, 0, stream>>>(
      stl::bind_back(scalar_cast<const Fn&>(scalars[0]),
                     scalar_cast<const Ss&>(scalars[SIs + 1])...),
      Red::policy::physical_view(  //
        as_mdspan_reduction<Fn, Dim>(reduction, working_set)),
      Is::policy::physical_view(
        as_mdspan<const stl::value_type_of_t<Is>, stl::dim_of_v<Is>>(inputs[IIs]))...,
      Os::policy::physical_view(
        as_mdspan<stl::value_type_of_t<Os>, stl::dim_of_v<Os>>(outputs[OIs]))...);
  }

  template <std::int32_t ActualDim>
  void operator()(std::vector<PhysicalArray>& reductions,
                  const std::vector<PhysicalArray>& inputs,
                  std::vector<PhysicalArray>& outputs,
                  const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t Dim = dim_of_v<Red>;

    if constexpr (Dim == ActualDim) {
      const Legion::Rect<Dim> shape = reductions.at(0).shape<Dim>();

      if (!shape.empty()) {
        impl(std::index_sequence_for<Is...>(),
             std::index_sequence_for<Os...>(),
             std::index_sequence_for<Ss...>(),
             reductions.at(0),
             inputs,
             outputs,
             scalars);
      }
    }
  }
};

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// reduction operation wrapper implementation
template <typename Reduction,
          typename Inputs,
          typename Outputs,
          typename Scalars,
          typename Constraints>
struct reduction_operation
  : LegateTask<reduction_operation<Reduction, Inputs, Outputs, Scalars, Constraints>> {
  static void cpu_variant(TaskContext context)
  {
    const auto inputs  = context.inputs();
    auto outputs       = context.outputs();
    const auto scalars = context.scalars();
    auto reductions    = context.reductions();

    dim_dispatch(reductions.at(0).dim(),
                 reduction_cpu<Reduction, Inputs, Outputs, Scalars>(),
                 reductions,
                 inputs,
                 outputs,
                 scalars);
  }

#if LegateDefined(LEGATE_USE_CUDA) && LegateDefined(REALM_COMPILER_IS_NVCC)
  // FIXME(wonchanl): In case where this template is instantiated multiple times with the exact same
  // template arguments, the exact class definition changes depending on what compiler is compiling
  // this header, which could lead to inconsistent class definitions across compile units.
  // Unfortunately, the -Wall flag doesn't allow us to have a member declaration having no
  // definition, so we can't fix the problem simply by pre-declaring this member all the time. The
  // right fix for this is to allow task variants to be defined in separate classes, instead of
  // requiring them to be members of the same class.
  static void gpu_variant(TaskContext context);
  {
    const auto inputs  = context.inputs();
    auto outputs       = context.outputs();
    const auto scalars = context.scalars();
    auto reductions    = context.reductions();

    dim_dispatch(reductions.at(0).dim(),
                 reduction_gpu<Reduction, Inputs, Outputs, Scalars>(),
                 reductions,
                 inputs,
                 outputs,
                 scalars);
  }
#endif
};

// _is_which is needed to disambiguate between the two overloads of
// get_arg below for gcc-9.
template <template <typename...> typename Which, typename What>
inline constexpr bool _is_which = false;

template <template <typename...> typename Which, typename... Args>
inline constexpr bool _is_which<Which, Which<Args...>> = true;

template <template <typename...> typename Which, typename... Ts, typename... Tail>
inline const Which<Ts...>& get_arg(const Which<Ts...>& head, const Tail&...)
{
  return head;
}

template <template <typename...> typename Which,
          typename Head,
#if 1  // LEGATE_STL_GCC() && (__GNUC__ <= 9)
          std::enable_if_t<!_is_which<Which, Head>, int> Enable = 0,
#endif
          typename... Tail>
inline decltype(auto) get_arg(const Head&, const Tail&... tail)
{
  return get_arg<Which>(tail...);
}

struct launch_task {
  template <typename Function,
            typename Inputs,
            typename Outputs,
            typename Scalars,
            typename Constraints>
  static auto make_iteration_task(
    Function function, Inputs inputs, Outputs outputs, Scalars scalars, Constraints constraints)
  {
    observer_ptr<Runtime> runtime = Runtime::get_runtime();
    Library library = runtime->find_or_create_library("legate.stl", LEGATE_STL_RESOURCE_CONFIG);
    const std::int64_t task_id =
      task_id_for<iteration_operation<Function, Inputs, Outputs, Scalars, Constraints>>(library);
    legate::AutoTask task = runtime->create_task(library, task_id);

    inputs(task, iteration_kind());
    outputs(task, iteration_kind());
    function(task);  // must preceed scalars
    scalars(task);
    constraints(task, inputs.data, outputs.data, inputs.data[0]);

    runtime->submit(std::move(task));
  }

  template <typename Reduction,
            typename Inputs,
            typename Outputs,
            typename Scalars,
            typename Constraints>
  static auto make_reduction_task(
    Reduction reduction, Inputs inputs, Outputs outputs, Scalars scalars, Constraints constraints)
  {
    observer_ptr<Runtime> runtime = Runtime::get_runtime();
    Library library = runtime->find_or_create_library("legate.stl", LEGATE_STL_RESOURCE_CONFIG);
    const std::int64_t task_id =
      task_id_for<reduction_operation<Reduction, Inputs, Outputs, Scalars, Constraints>>(library);
    legate::AutoTask task = runtime->create_task(library, task_id);

    inputs(task, reduction_kind());
    outputs(task, reduction_kind());
    reduction(task);  // must preceed scalars
    scalars(task);
    constraints(task, inputs.data, outputs.data, reduction.data);

    runtime->submit(std::move(task));
  }

  template <typename... Ts>
  auto operator()(Ts... args) const
  {
    detail::inputs<> no_inputs;
    detail::outputs<> no_outputs;
    detail::scalars<> no_scalars;
    detail::function<> no_function;
    detail::reduction<> no_reduction;
    detail::constraints<> no_constraints;

    auto function  = detail::get_arg<detail::function>(args..., no_function);
    auto reduction = detail::get_arg<detail::reduction>(args..., no_reduction);

    constexpr bool has_function  = !std::is_same_v<decltype(function), detail::function<>>;
    constexpr bool has_reduction = !std::is_same_v<decltype(reduction), detail::reduction<>>;

    static_assert((has_function + has_reduction) == 1,
                  "You must specify either a function or a reduction");

    if constexpr (has_function) {
      return make_iteration_task(function,
                                 detail::get_arg<detail::inputs>(args..., no_inputs),
                                 detail::get_arg<detail::outputs>(args..., no_outputs),
                                 detail::get_arg<detail::scalars>(args..., no_scalars),
                                 detail::get_arg<detail::constraints>(args..., no_constraints));
    } else if constexpr (has_reduction) {
      return make_reduction_task(reduction,
                                 detail::get_arg<detail::inputs>(args..., no_inputs),
                                 detail::get_arg<detail::outputs>(args..., no_outputs),
                                 detail::get_arg<detail::scalars>(args..., no_scalars),
                                 detail::get_arg<detail::constraints>(args..., no_constraints));
    }
  }
};

}  // namespace detail

inline constexpr detail::make_inputs inputs{};
inline constexpr detail::make_outputs outputs{};
inline constexpr detail::make_scalars scalars{};
inline constexpr detail::make_function function{};
inline constexpr detail::make_constraints constraints{};
inline constexpr detail::make_reduction reduction{};

inline constexpr detail::make_align align{};
// TODO: broadcasting

inline constexpr detail::launch_task launch_task{};

}  // namespace legate::stl

#include "suffix.hpp"
