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
#include "utility.hpp"

#include <atomic>
#include <iterator>
#include <vector>

// Include this last:
#include "prefix.hpp"

namespace legate::stl {

namespace detail {

/**
 * @brief A class representing a collection of inputs for a task.
 *
 * This class is used to specify the inputs for a task in the Legate runtime system.
 * It holds a list of logical stores and provides methods to apply the inputs to a task.
 */
template <typename... Ts>
class inputs {
 private:
  using Types = meta::list<Ts...>;
  std::vector<LogicalStore> data_{};

  /**
   * @brief Apply the inputs to a task.
   *
   * This method adds the logical stores as inputs to the given task.
   * It also adds partitioning constraints based on the input types.
   *
   * @param task The task to apply the inputs to.
   * @param kind The kind of the task (iteration or reduction).
   * @param index The index of the input.
   */
  template <typename Type, typename Kind>
  void apply(AutoTask& task, Kind kind, std::size_t index) const
  {
    // Add the stores as inputs to the task
    Variable part = task.find_or_declare_partition(data()[index]);
    task.add_input(data()[index], part);

    // Add the partitioning constraints
    auto constraints = Type::policy::partition_constraints(kind);
    std::apply([&](auto&&... cs) { (task.add_constraint(cs(part)), ...); }, std::move(constraints));
  }

 public:
  inputs() = default;

  explicit inputs(std::vector<LogicalStore> data) : data_{std::move(data)} {}

  /**
   * @brief Apply the inputs to a task.
   *
   * This method applies the inputs to the given task.
   * It iterates over the inputs and calls the apply method for each input type.
   *
   * @param task The task to apply the inputs to.
   * @param kind The kind of the task  (iteration or reduction).
   */
  template <typename Kind>
  void operator()(AutoTask& task, [[maybe_unused]] Kind kind) const
  {
    LegateAssert(data().size() == sizeof...(Ts));

    std::size_t index = 0;
    (this->apply<Ts>(task, kind, index++), ...);
  }

  [[nodiscard]] const std::vector<LogicalStore>& data() const noexcept { return data_; }
};

/**
 * @brief A class representing the outputs of a task.
 *
 * This class is used to specify the outputs of a task in the Legate framework.
 * It holds a list of logical stores and provides methods to apply the outputs
 * to a task and add partitioning constraints.
 *
 */
template <typename... Ts>
class outputs {
 private:
  using Types = meta::list<Ts...>;
  std::vector<LogicalStore> data_{};

  /**
   * @brief Apply the outputs to a task.
   *
   * This method adds the logical stores as outputs to the given task and adds
   * partitioning constraints based on the output types.
   *
   * @param task The task to apply the outputs to.
   * @param kind The kind of the task (iteration or reduction).
   * @param index The index of the output.
   */
  template <typename Type, typename Kind>
  void apply(AutoTask& task, Kind kind, std::size_t index) const
  {
    // Add the stores as outputs to the task
    Variable part = task.find_or_declare_partition(data()[index]);
    task.add_output(data()[index], part);

    // Add the partitioning constraints
    auto constraints = Type::policy::partition_constraints(kind);
    std::apply([&](auto&&... cs) { (task.add_constraint(cs(part)), ...); }, std::move(constraints));
  }

 public:
  outputs() = default;

  explicit outputs(std::vector<LogicalStore> data) : data_{std::move(data)} {}

  /**
   * @brief Apply the outputs to a task.
   *
   * This method applies the outputs to the given task by calling the `apply`
   * method for each output type.
   *
   * @param task The task to apply the outputs to.
   * @param kind The kind of the task (iteration or reduction).
   */
  template <typename Kind>
  void operator()(AutoTask& task, [[maybe_unused]] Kind kind) const
  {
    LegateAssert(data().size() == sizeof...(Ts));

    std::size_t index = 0;
    (this->apply<Ts>(task, kind, index++), ...);
  }

  [[nodiscard]] const std::vector<LogicalStore>& data() const noexcept { return data_; }
};

/**
 * @brief A class representing a set of constraints for a task.
 *
 * This class holds a tuple of constraints that can be applied to a task.
 * When the constraints are invoked, they are passed the task, input logical stores,
 * output logical stores, and a reduction logical store.
 */
template <typename... Ts>
class constraints {
 public:
  std::tuple<Ts...> data{};

  /**
   * @brief Invoke the constraints on a task.
   *
   * This function applies each constraint in the tuple to the given task,
   * input logical stores, output logical stores, and reduction logical store.
   *
   * @param task The task to apply the constraints to.
   * @param inputs The input logical stores as a `std::vector<LogicalStore>`.
   * @param outputs The output logical stores as a `std::vector<LogicalStore>`.
   * @param reduction The reduction logical store.
   */
  void operator()(AutoTask& task,
                  const std::vector<LogicalStore>& inputs,
                  const std::vector<LogicalStore>& outputs,
                  const LogicalStore& reduction) const
  {
    std::apply([&](auto&&... cons) { (cons(task, inputs, outputs, reduction), ...); }, data);
  }
};

/**
 * @brief A class template representing a collection of scalar values.
 *
 * This class template is used to store a tuple of scalar values and provide
 * a callable operator to add them as arguments to an `AutoTask` object.
 *
 */
template <typename... Ts>
class scalars {
 public:
  std::tuple<Ts...> data{};

  /**
   * @brief Adds the scalar values as arguments to the given `AutoTask` object.
   *
   * This operator function adds each scalar value in the `data` tuple as an argument
   * to the provided `AutoTask` object. It uses `std::apply` to iterate over the tuple
   * and invoke the lambda function that adds each scalar value as an argument.
   *
   * @param task The `AutoTask` object to which the scalar values will be added as arguments.
   */
  void operator()(AutoTask& task) const
  {
    std::apply(
      [&](auto&... scalar) {
        (task.add_scalar_arg(Scalar{binary_type(sizeof(scalar)), std::addressof(scalar), true}),
         ...);
      },
      data);
  }
};

template <typename... Fn>
class function;

/**
 * @cond
 * This specialization is used as an implementation detail of the `launch_task`
 * function.
 */
template <>
class function<> {};
/**
 * @endcond
 */

/**
 * @brief A class template representing a (possibly stateful) function object.
 *
 * This class template is used to store a function object and provide
 * a callable operator to add it as a scalar argument to an `AutoTask` object.
 *
 */
template <typename Fn>
class function<Fn> {
 public:
  Fn fn{};

  /**
   * @brief Adds the function as a scalar value argument to the given `AutoTask` object.
   *
   * This function is responsible for executing the provided task.
   *
   * @param task The task to be executed.
   */
  void operator()(AutoTask& task) const
  {
    task.add_scalar_arg(Scalar{binary_type(sizeof(fn)), std::addressof(fn), true});
  }
};

/**
 * @cond
 */
[[nodiscard]] inline std::int32_t _next_reduction_id()
{
  static std::atomic<std::int32_t> id{LEGION_REDOP_LAST};
  return id.fetch_add(1);
}

template <typename T>
[[nodiscard]] std::int32_t _reduction_id_for()
{
  static const std::int32_t id = _next_reduction_id();
  return id;
}

[[nodiscard]] inline std::int32_t _next_reduction_kind()
{
  static std::atomic<std::int32_t> id{static_cast<std::int32_t>(ReductionOpKind::XOR) + 1};
  return id.fetch_add(1);
}

template <typename T>
[[nodiscard]] std::int32_t _reduction_kind_for()
{
  static const std::int32_t id = _next_reduction_kind();
  return id;
}

template <typename Fun>
[[nodiscard]] std::int32_t _get_reduction_id()
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
[[nodiscard]] std::int32_t _record_reduction_for()
{
  static const auto KIND = []() -> std::int32_t {
    const Type type{primitive_type(type_code_of<ElementType>)};
    const std::int32_t id   = _get_reduction_id<Fun>();
    const std::int32_t kind = _reduction_kind_for<Fun>();

    type.record_reduction_operator(kind, id);
    return kind;
  }();
  return KIND;
}
/**
 * @endcond
 */

template <typename...>
class reduction;

/**
 * @cond
 * This specialization is used as an implementation detail of the `launch_task`
 * function.
 */
template <>
class reduction<> {};
/**
 * @endcond
 */

/**
 * @brief Class template for reduction operations.
 *
 * This class template represents a reduction operation for a Legate task.
 * It stores a logical store and the reduction function and provides
 * a callable operator to add it as a reduction and a scalar argument to an `AutoTask` object.
 */
template <typename Store, typename Fun>
class reduction<Store, Fun> {
 public:
  LogicalStore data{}; /**< The logical store on which the reduction operation is performed. */
  Fun fn{};            /**< The reduction function to be applied. */

  /**
   * @brief Function call operator for the reduction operation.
   *
   * This function adds the reduction operation to an `AutoTask`.
   * It finds or declares the partition for the logical store, records the reduction operation,
   * and adds the reduction function as a scalar arguments to the task.
   *
   * @param task The AutoTask on which the reduction operation is invoked.
   */
  void operator()(AutoTask& task) const
  {
    auto part       = task.find_or_declare_partition(data);
    const auto kind = _record_reduction_for<element_type_of_t<Store>, Fun>();

    task.add_reduction(data, kind, std::move(part));
    task.add_scalar_arg(Scalar{binary_type(sizeof(fn)), std::addressof(fn), true});
  }
};

/**
 * @cond
 */
enum class store_type : int { input, output, reduction };

class store_placeholder {
 public:
  store_type which{};
  int index{};

 private:
  template <typename, typename>
  friend class align;

  [[nodiscard]] LogicalStore operator()(const std::vector<LogicalStore>& inputs,
                                        const std::vector<LogicalStore>& outputs,
                                        const LogicalStore& reduction) const
  {
    switch (which) {
      case store_type::input: return inputs[index];
      case store_type::output: return outputs[index];
      case store_type::reduction: return reduction;
    }
    // src/stl/detail/launch_task.hpp:238:3: error: control reaches end of non-void function
    // [-Werror=return-type]
    //
    // ... I mean, it doesn't, since that switch above is fully covered...
    LegateUnreachable();
  }
};

template <typename Left, typename Right>
class align {
 public:
  align(Left left, Right right) : left_{std::move(left)}, right_{std::move(right)} {}

  void operator()(AutoTask& task,
                  const std::vector<LogicalStore>& inputs,
                  const std::vector<LogicalStore>& outputs,
                  const LogicalStore& reduction) const
  {
    do_align(task, left_(inputs, outputs, reduction), right_(inputs, outputs, reduction));
  }

 private:
  static void do_align(AutoTask& task, Variable left, Variable right)
  {
    if (left.impl() != right.impl()) {
      task.add_constraint(legate::align(left, right));
    }
  }

  static void do_align(AutoTask& task, const LogicalStore& left, const LogicalStore& right)
  {
    do_align(task, task.find_or_declare_partition(left), task.find_or_declare_partition(right));
  }

  static void do_align(AutoTask& task,
                       const LogicalStore& left,
                       const std::vector<LogicalStore>& right)
  {
    auto left_part = task.find_or_declare_partition(left);
    for (auto& store : right) {
      do_align(task, left_part, task.find_or_declare_partition(store));
    }
  }

  static void do_align(AutoTask& task,
                       const std::vector<LogicalStore>& left,
                       const LogicalStore& right)
  {
    auto right_part = task.find_or_declare_partition(right);
    for (auto& store : left) {
      do_align(task, task.find_or_declare_partition(store), right_part);
    }
  }

  Left left_{};
  Right right_{};
};

class make_inputs {
 public:
  template <typename... Ts>                  //
    requires(logical_store_like<Ts> && ...)  //
  [[nodiscard]] inputs<std::remove_reference_t<Ts>...> operator()(Ts&&... stores) const
  {
    return inputs<std::remove_reference_t<Ts>...>{
      std::vector<LogicalStore>{get_logical_store(std::forward<Ts>(stores))...}};
  }

  [[nodiscard]] store_placeholder operator[](int index) const { return {store_type::input, index}; }

 private:
  template <typename, typename>
  friend class align;

  [[nodiscard]] const std::vector<LogicalStore>& operator()(
    const std::vector<LogicalStore>& inputs,
    const std::vector<LogicalStore>& /*outputs*/,
    const LogicalStore& /*reduction*/) const
  {
    return inputs;
  }
};

class make_outputs {
 public:
  template <typename... Ts>                  //
    requires(logical_store_like<Ts> && ...)  //
  [[nodiscard]] outputs<std::remove_reference_t<Ts>...> operator()(Ts&&... stores) const
  {
    return outputs<std::remove_reference_t<Ts>...>{
      std::vector<LogicalStore>{get_logical_store(std::forward<Ts>(stores))...}};
  }

  [[nodiscard]] store_placeholder operator[](int index) const
  {
    return {store_type::output, index};
  }

 private:
  template <typename, typename>
  friend class align;

  [[nodiscard]] const std::vector<LogicalStore>& operator()(
    const std::vector<LogicalStore>& /*inputs*/,
    const std::vector<LogicalStore>& outputs,
    const LogicalStore& /*reduction*/) const
  {
    return outputs;
  }
};

class make_scalars {
 public:
  template <typename... Ts>
  [[nodiscard]] scalars<Ts...> operator()(Ts&&... scalars) const
  {
    static_assert((std::is_trivially_copyable_v<std::decay_t<Ts>> && ...),
                  "All scalar arguments must be trivially copyable");
    return {{std::forward<Ts>(scalars)...}};
  }
};

class make_function {
 public:
  template <typename Fn>
  [[nodiscard]] function<std::decay_t<Fn>> operator()(Fn&& fn) const
  {
    return {std::forward<Fn>(fn)};
  }
};

template <typename Store>
[[nodiscard]] constexpr std::int32_t _dim_of() noexcept
{
  return dim_of_v<Store>;
}

template <typename Store>
using dim_of_t = meta::constant<_dim_of<Store>()>;

class make_reduction {
 public:
  template <typename Store, typename ReductionFn>  //
    requires(logical_store_like<Store>)            // TODO(ericniebler): constrain Fun
  [[nodiscard]] reduction<std::remove_reference_t<Store>, std::decay_t<ReductionFn>> operator()(
    Store&& store, ReductionFn&& reduction) const
  {
    static_assert(legate_reduction<ReductionFn>,
                  "The stl::reduction() function requires a Legate reduction operation "
                  "such as legate::SumReduction or legate::MaxReduction");
    return {get_logical_store(std::forward<Store>(store)), std::forward<ReductionFn>(reduction)};
  }

 private:
  template <typename, typename>
  friend class align;

  [[nodiscard]] LogicalStore operator()(const std::vector<LogicalStore>& /*inputs*/,
                                        const std::vector<LogicalStore>& /*outputs*/,
                                        const LogicalStore& reduction) const
  {
    return reduction;
  }
};

class make_constraints {
 public:
  template <typename... Ts>  //
    requires((callable<Ts,
                       AutoTask&,
                       const std::vector<LogicalStore>&,
                       const std::vector<LogicalStore>&,
                       const LogicalStore&> &&
              ...))
  [[nodiscard]] constraints<std::decay_t<Ts>...> operator()(Ts&&... constraints) const
  {
    return {{std::forward<Ts>(constraints)...}};
  }
};

class make_align {
 public:
  // E.g., `align(inputs[0], inputs[1])`
  //       `align(outputs[0], inputs)`
  template <typename Left, typename Right>
    requires(callable<Left,
                      const std::vector<LogicalStore>&,
                      const std::vector<LogicalStore>&,
                      const LogicalStore&> &&
             callable<Right,
                      const std::vector<LogicalStore>&,
                      const std::vector<LogicalStore>&,
                      const LogicalStore&>)  //
  [[nodiscard]] align<Left, Right> operator()(Left left, Right right) const
  {
    return {left, right};
  }

  // For `align(inputs)`
  [[nodiscard]] auto operator()(make_inputs inputs) const { return (*this)(inputs[0], inputs); }

  // For `align(outputs)`
  [[nodiscard]] auto operator()(make_outputs outputs) const { return (*this)(outputs[0], outputs); }
};

template <typename Fn, typename... Views>
void _cpu_for_each(Fn fn, Views... views)
{
  auto&& input0 = front_of(views...);
  auto&& begin  = input0.begin();

  static_assert_iterator_category<std::forward_iterator_tag>(begin);

  const auto distance = std::distance(std::move(begin), input0.end());

  for (std::int64_t idx = 0; idx < distance; ++idx) {  //
    fn(*(views.begin() + idx)...);
  }
}

template <typename Function, typename inputs, typename Outputs, typename Scalars>
class iteration_cpu;

// This is a CPU implementation of a for_each operation.
template <typename Fn, typename... Is, typename... Os, typename... Ss>
class iteration_cpu<function<Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
 public:
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
        impl(std::index_sequence_for<Is...>{},
             std::index_sequence_for<Os...>{},
             std::index_sequence_for<Ss...>{},
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
  const auto idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto&& input0  = front_of(views...);

  static_assert_iterator_category<std::forward_iterator_tag>(input0.begin());

  const auto distance = input0.end() - input0.begin();

  if (idx < distance) {  //
    fn(*(views.begin() + idx)...);
  }
}

template <typename Function, typename inputs, typename Outputs, typename Scalars>
class iteration_gpu;

// This is a GPU implementation of a for_each operation.
template <typename Fn, typename... Is, typename... Os, typename... Ss>
class iteration_gpu<function<Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
 public:
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
        impl(std::index_sequence_for<Is...>{},
             std::index_sequence_for<Os...>{},
             std::index_sequence_for<Ss...>{},
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
    auto&& inputs  = context.inputs();
    auto&& outputs = context.outputs();
    auto&& scalars = context.scalars();
    const auto dim = inputs.at(0).dim();

    dim_dispatch(
      dim, iteration_cpu<Function, Inputs, Outputs, Scalars>{}, inputs, outputs, scalars);
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
    auto&& inputs  = context.inputs();
    auto&& outputs = context.outputs();
    auto&& scalars = context.scalars();
    const auto dim = inputs.at(0).dim();

    dim_dispatch(
      dim, iteration_gpu<Function, Inputs, Outputs, Scalars>{}, inputs, outputs, scalars);
  }
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename Op, std::int32_t Dim, bool Exclusive = false>
[[nodiscard]] inline auto as_mdspan_reduction(PhysicalArray& array, Rect<Dim> working_set)
  -> mdspan_reduction_t<Op, Dim, Exclusive>
{
  PhysicalStore store = array.data();

  using Mapping = std::layout_right::mapping<std::dextents<coord_t, Dim>>;
  Mapping mapping{detail::dynamic_extents<Dim>(working_set)};

  using Policy   = reduction_accessor<Op, Exclusive>;
  using Accessor = detail::mdspan_accessor<typename Op::RHS, Dim, Policy>;
  Accessor accessor{std::move(store), std::move(working_set)};

  using Handle = typename Accessor::data_handle_type;
  Handle handle{};

  return {std::move(handle), std::move(mapping), std::move(accessor)};
}

template <typename Function, typename InOut, typename Input>
void _cpu_reduce(Function fn, InOut inout, Input input)
{
  // These need to be at least multi-pass
  static_assert_iterator_category<std::forward_iterator_tag>(inout.begin());
  static_assert_iterator_category<std::forward_iterator_tag>(input.begin());
  const auto distance = std::distance(inout.begin(), inout.end());

  LegateAssert(distance == std::distance(input.begin(), input.end()));
  for (std::int64_t idx = 0; idx < distance; ++idx) {
    fn(*(inout.begin() + idx), *(input.begin() + idx));
  }
}

template <typename Reduction, typename Inputs, typename Outputs, typename Scalars>
class reduction_cpu;

template <typename Red, typename Fn, typename... Is, typename... Os, typename... Ss>
class reduction_cpu<reduction<Red, Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
 public:
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
                  as_mdspan_reduction<Fn, Dim>(reduction, std::move(working_set))),
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
        impl(std::index_sequence_for<Is...>{},
             std::index_sequence_for<Os...>{},
             std::index_sequence_for<Ss...>{},
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
  const auto tid      = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto distance = inout.end() - inout.begin();

  LegateAssert(distance == (input.end() - input.begin()));
  for (std::int64_t idx = 0; idx < distance; ++idx) {
    fn(tid, *(inout.begin() + idx), *(input.begin() + idx));
  }
}

template <typename Reduction, typename Inputs, typename Outputs, typename Scalars>
class reduction_gpu;

template <typename Red, typename Fn, typename... Is, typename... Os, typename... Ss>
class reduction_gpu<reduction<Red, Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
 public:
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
        impl(std::index_sequence_for<Is...>{},
             std::index_sequence_for<Os...>{},
             std::index_sequence_for<Ss...>{},
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
    auto&& inputs     = context.inputs();
    auto&& outputs    = context.outputs();
    auto&& scalars    = context.scalars();
    auto&& reductions = context.reductions();
    const auto dim    = reductions.at(0).dim();

    dim_dispatch(dim,
                 reduction_cpu<Reduction, Inputs, Outputs, Scalars>{},
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
    auto&& inputs     = context.inputs();
    auto&& outputs    = context.outputs();
    auto&& scalars    = context.scalars();
    auto&& reductions = context.reductions();
    const auto dim    = reductions.at(0).dim();

    dim_dispatch(dim,
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
[[nodiscard]] inline const Which<Ts...>& get_arg(const Which<Ts...>& head, const Tail&...)
{
  return head;
}

template <template <typename...> typename Which,
          typename Head,
          std::enable_if_t<!_is_which<Which, Head>, int> Enable = 0,
          typename... Tail>
[[nodiscard]] inline decltype(auto) get_arg(const Head&, const Tail&... tail)
{
  return get_arg<Which>(tail...);
}
/**
 * @endcond
 */

/**
 * @class launch_task
 * @brief A class that represents a task launcher.
 *
 * The `launch_task` class provides a convenient interface for launching tasks in the Legate
 * framework. It supports both iteration tasks and reduction tasks. The tasks are created and
 * submitted to the runtime using the provided inputs, outputs, scalars, and constraints.
 */
class launch_task {
  template <typename LegateTask>
  [[nodiscard]] static std::tuple<legate::AutoTask, observer_ptr<Runtime>> make_task_()
  {
    const auto runtime = Runtime::get_runtime();
    auto library       = runtime->find_or_create_library("legate.stl", LEGATE_STL_RESOURCE_CONFIG);
    const auto task_id = task_id_for<LegateTask>(library);

    return {runtime->create_task(std::move(library), task_id), runtime};
  }

  /**
   * @brief Creates an iteration task with the given function, inputs, outputs, scalars, and
   * constraints.
   *
   * This function creates an iteration task using the provided function, inputs, outputs, scalars,
   * and constraints. It retrieves the runtime and library, and then creates the task using a unique
   * task ID for the iteration operation. The inputs and outputs are set for the task, followed by
   * the function and scalars. Finally, the constraints are set using the input and output data, and
   * the task is submitted to the runtime.
   *
   * @param function The function to be executed in the task.
   * @param inputs The inputs for the task.
   * @param outputs The outputs for the task.
   * @param scalars The scalars for the task.
   * @param constraints The constraints for the task.
   */
  template <typename Function,
            typename Inputs,
            typename Outputs,
            typename Scalars,
            typename Constraints>
  static void make_iteration_task(
    Function function, Inputs inputs, Outputs outputs, Scalars scalars, Constraints constraints)
  {
    auto&& [task, runtime] =
      make_task_<iteration_operation<Function, Inputs, Outputs, Scalars, Constraints>>();

    inputs(task, iteration_kind{});
    outputs(task, iteration_kind{});
    function(task);  // must preceed scalars
    scalars(task);
    constraints(task, inputs.data(), outputs.data(), inputs.data()[0]);

    runtime->submit(std::move(task));
  }

  /**
   * @brief Creates a reduction task with the given inputs, outputs, scalars, and constraints.
   *
   * This function creates an iteration task using the provided function, inputs, outputs, scalars,
   * and constraints. It retrieves the runtime and library, and then creates the task using a unique
   * task ID for the iteration operation. The inputs and outputs are set for the task, followed by
   * the function and scalars. Finally, the constraints are set using the input and output data, and
   * the task is submitted to the runtime.
   *
   * @param reduction The reduction operation.
   * @param inputs The inputs.
   * @param outputs The outputs.
   * @param scalars The scalars.
   * @param constraints The constraints.
   */
  template <typename Reduction,
            typename Inputs,
            typename Outputs,
            typename Scalars,
            typename Constraints>
  static void make_reduction_task(
    Reduction reduction, Inputs inputs, Outputs outputs, Scalars scalars, Constraints constraints)
  {
    auto&& [task, runtime] =
      make_task_<reduction_operation<Reduction, Inputs, Outputs, Scalars, Constraints>>();

    inputs(task, reduction_kind{});
    outputs(task, reduction_kind{});
    reduction(task);  // must preceed scalars
    scalars(task);
    constraints(task, inputs.data(), outputs.data(), reduction.data);

    runtime->submit(std::move(task));
  }

 public:
  /**
   * @brief Launches a task with specified arguments.
   *
   * This function template is used to launch a task with specified arguments. It supports both
   * iteration tasks and reduction tasks.
   *
   * @param args The arguments for the task.
   *
   * @requires Either a `function<>` or a `reduction<>` argument must be specified.
   */
  template <typename... Ts>
  void operator()(Ts... args) const
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
      make_iteration_task(std::move(function),
                          detail::get_arg<detail::inputs>(args..., no_inputs),
                          detail::get_arg<detail::outputs>(args..., no_outputs),
                          detail::get_arg<detail::scalars>(args..., no_scalars),
                          detail::get_arg<detail::constraints>(args..., no_constraints));
    } else if constexpr (has_reduction) {
      make_reduction_task(std::move(reduction),
                          detail::get_arg<detail::inputs>(args..., no_inputs),
                          detail::get_arg<detail::outputs>(args..., no_outputs),
                          detail::get_arg<detail::scalars>(args..., no_scalars),
                          detail::get_arg<detail::constraints>(args..., no_constraints));
    } else {
      static_assert(has_function || has_reduction,
                    "You must specify either a function or a reduction");
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
// TODO(ericniebler): broadcasting

/**
 * @cond
 */
inline constexpr detail::launch_task launch_task{};
/**
 * @endcond
 */

#if LegateDefined(LEGATE_STL_DOXYGEN)
/**
 * @brief A function that launches a task with the given inputs, outputs,
 * scalars, and constraints.
 *
 * Launch parameter arguments can be one of the following in any order:
 *
 * - `legate::stl::inputs` - specifies the input stores for the task
 *    - \a Example:
 *
 *      @code
 *      inputs(store1, store2, store3)
 *      @endcode
 *
 * - `legate::stl::outputs` - specifies the output stores for the task
 *    - \a Example:
 *
 *      @code
 *      outputs(store1, store2, store3)
 *      @endcode
 *
 * - `legate::stl::scalars` - specifies the scalar arguments for the task
 *    - \a Example:
 *
 *      @code
 *      scalars(42, 3.14f)
 *      @endcode
 *
 * - `legate::stl::function` - specifies the function to be applied
 *    iteratively to the inputs.
 *    - The function will take as arguments the current elements of the
 *      input stores, in order, followed by the current elements of the
 *      output stores. The elements of a `stl::logical_store` are lvalue
 *      references to the elements of the physical store it represents.
 *      The elements of a view such as `stl::rows_of(store)` are `mdspan`s
 *      denoting the rows of `store`.
 *    - The function must be bitwise copyable.
 *    - Only one of `function` or `reduction` can be specified in a call
 *      to `launch_task`
 *    - \a Example:
 *
 *      @code{cpp}
 *      function([](auto& in, const auto& out) { in = out * out; })
 *      @endcode
 *
 * - `legate::stl::reduction` - specifies the reduction store and the
 *    reduction function to be applied to the inputs.
 *    - The function must be bitwise copyable.
 *    - The function must take as arguments a reduction store and a binary
 *      reduction function.
 *    - The reduction store can be a `logical_store` or some view of a
 *      store, such as `rows_of(store)`. When operating on a view, the
 *      arguments to the reduction function will be the elements of the
 *      view. For example, if the reduction store is `rows_of(store)`,
 *      the arguments passed to the reduction function will be `mdspan`s
 *      denoting rows of `store`.
 *    - Only one of `function` or `reduction` can be specified in a call
 *      to `launch_task`
 *    - \a Example:
 *
 *      @code{cpp}
 *      stl::reduction(stl::rows_of(store), stl::elementwise(std::plus{}))
 *      @endcode
 *
 * - `legate::stl::constraints` - specifies the constraints for the task
 *    - A constraint is a callable that takes an `legate::AutoTask&` and
 *      the input, output, and reduction stores as arguments. Its function
 *      signature must be:
 *
 *      @code{cpp}
 *      void(legate::AutoTask&,                // the task to add the constraints to
 *           const std::vector<LogicalStore>&, // the input stores
 *           const std::vector<LogicalStore>&, // the output stores
 *           const LogicalStore&)              // the reduction store
 *      @endcode
 *
 *    - Legate provides one constraint generator, `legate::stl::align`, for specifying
 *      the alignment constraints for the task. It can be used many different ways:
 *       - `align(inputs[0], inputs[1])` - aligns the first input with the second input
 *       - `align(inputs[0], outputs[0])` - aligns the first input with the first output
 *       - `align(outputs[0], inputs)` - aligns the first output with all the inputs
 *       - `align(outputs, inputs[1])` - aligns all the outputs with the second input
 *       - `align(reduction, inputs[0])` - aligns the reduction store with the first input
 *       - `align(reduction, inputs)` - aligns the reduction store with all the input
 *       - `align(inputs)` - aligns all the inputs with each other
 *       - `align(outputs)` - aligns all the outputs with each other
 *
 * @par Example
 * The following example shows how to use `launch_task` to implement a `for_each`
 * algorithm that iterates over two input stores.
 *
 * @code{cpp}
 * template <class Function, class Input1, class Input2>
 * void for_each_zip(Function fn, Input1&& input1, Input2&& input2) {
 *   auto drop_inputs = [fn](const auto&, const auto&, auto& out1, auto& out2) {
 *     fn(out1, out2);
 *   };
 *   legate::stl::launch_task(
 *     legate::stl::inputs(input1, input2),
 *     legate::stl::outputs(input1, input2),
 *     legate::stl::function(drop_inputs),
 *     legate::stl::constraints(legate::stl::align(input1, input2)));
 * }
 * @endcode
 *
 */
template <LaunchParam... Params>
void launch_task(Params... params);
#endif

}  // namespace legate::stl

#include "suffix.hpp"
