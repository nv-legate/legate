/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/experimental/stl/detail/functional.hpp>
#include <legate/experimental/stl/detail/meta.hpp>
#include <legate/experimental/stl/detail/registrar.hpp>
#include <legate/experimental/stl/detail/slice.hpp>
#include <legate/experimental/stl/detail/store.hpp>
#include <legate/experimental/stl/detail/utility.hpp>
#include <legate/task/task_config.h>
#include <legate/utilities/assert.h>
#include <legate/utilities/macros.h>

#include <atomic>
#include <iterator>
#include <vector>

// Include this last:
#include <legate/experimental/stl/detail/prefix.hpp>

namespace legate::experimental::stl {

namespace detail {

/**
 * @brief A class representing a collection of inputs for a task.
 *
 * This class is used to specify the inputs for a task in the Legate runtime system.
 * It holds a list of logical stores and provides methods to apply the inputs to a task.
 *
 * @ingroup stl-utilities
 */
template <typename... Ts>
class Inputs {
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
   *
   * @ingroup stl-utilities
   */
  template <typename Type, typename Kind>
  void apply_(AutoTask& task, Kind kind, std::size_t index) const
  {
    // Add the stores as inputs to the task
    Variable part = task.find_or_declare_partition(data()[index]);
    task.add_input(data()[index], part);

    // Add the partitioning constraints
    auto constraints = Type::policy::partition_constraints(kind);
    std::apply([&](auto&&... cs) { (task.add_constraint(cs(part)), ...); }, std::move(constraints));
  }

 public:
  Inputs() = default;

  explicit Inputs(std::vector<LogicalStore> data) : data_{std::move(data)} {}

  /**
   * @brief Apply the inputs to a task.
   *
   * This method applies the inputs to the given task.
   * It iterates over the inputs and calls the apply method for each input type.
   *
   * @param task The task to apply the inputs to.
   * @param kind The kind of the task  (iteration or reduction).
   *
   * @ingroup stl-utilities
   */
  template <typename Kind>
  void operator()(AutoTask& task, [[maybe_unused]] Kind kind) const
  {
    LEGATE_ASSERT(data().size() == sizeof...(Ts));

    std::size_t index = 0;
    (this->apply_<Ts>(task, kind, index++), ...);
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
 * @ingroup stl-utilities
 */
template <typename... Ts>
class Outputs {
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
   *
   * @ingroup stl-utilities
   */
  template <typename Type, typename Kind>
  void apply_(AutoTask& task, Kind kind, std::size_t index) const
  {
    // Add the stores as outputs to the task
    Variable part = task.find_or_declare_partition(data()[index]);
    task.add_output(data()[index], part);

    // Add the partitioning constraints
    auto constraints = Type::policy::partition_constraints(kind);
    std::apply([&](auto&&... cs) { (task.add_constraint(cs(part)), ...); }, std::move(constraints));
  }

 public:
  Outputs() = default;

  explicit Outputs(std::vector<LogicalStore> data) : data_{std::move(data)} {}

  /**
   * @brief Apply the outputs to a task.
   *
   * This method applies the outputs to the given task by calling the `apply`
   * method for each output type.
   *
   * @param task The task to apply the outputs to.
   * @param kind The kind of the task (iteration or reduction).
   *
   * @ingroup stl-utilities
   */
  template <typename Kind>
  void operator()(AutoTask& task, [[maybe_unused]] Kind kind) const
  {
    LEGATE_ASSERT(data().size() == sizeof...(Ts));

    // No, clang-tidy, index can *not* be marked as const
    std::size_t index = 0;  // NOLINT(misc-const-correctness)
    (this->apply_<Ts>(task, kind, index++), ...);
  }

  [[nodiscard]] const std::vector<LogicalStore>& data() const noexcept { return data_; }
};

/**
 * @brief A class representing a set of constraints for a task.
 *
 * This class holds a tuple of constraints that can be applied to a task.
 * When the constraints are invoked, they are passed the task, input logical stores,
 * output logical stores, and a reduction logical store.
 *
 * @ingroup stl-utilities
 */
template <typename... Ts>
class Constraints {
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
   *
   * @ingroup stl-utilities
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
 *
 * @ingroup stl-utilities
 */
template <typename... Ts>
class Scalars {
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
   *
   * @ingroup stl-utilities
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
class Function;

/**
 * @cond
 * This specialization is used as an implementation detail of the `launch_task`
 * function.
 */
template <>
class Function<> {};
/**
 * @endcond
 */

/**
 * @brief A class template representing a (possibly stateful) function object.
 *
 * This class template is used to store a function object and provide
 * a callable operator to add it as a scalar argument to an `AutoTask` object.
 *
 * @ingroup stl-utilities
 */
template <typename Fn>
class Function<Fn> {
 public:
  Fn fn{};

  /**
   * @brief Adds the function as a scalar value argument to the given `AutoTask` object.
   *
   * This function is responsible for executing the provided task.
   *
   * @param task The task to be executed.
   *
   * @ingroup stl-utilities
   */
  void operator()(AutoTask& task) const
  {
    task.add_scalar_arg(Scalar{binary_type(sizeof(fn)), std::addressof(fn), true});
  }
};

/**
 * @cond
 */
[[nodiscard]] inline LocalRedopID next_reduction_id_()  // NOLINT(readability-identifier-naming)
{
  static std::atomic<std::underlying_type_t<LocalRedopID>> id{};
  return static_cast<LocalRedopID>(id.fetch_add(1));
}

template <typename T>
[[nodiscard]] LocalRedopID reduction_id_for_()  // NOLINT(readability-identifier-naming)
{
  static const LocalRedopID id = next_reduction_id_();
  return id;
}

[[nodiscard]] inline std::int32_t next_reduction_kind_()  // NOLINT(readability-identifier-naming)
{
  static std::atomic<std::int32_t> id{legate::detail::to_underlying(ReductionOpKind::XOR) + 1};
  return id.fetch_add(1);
}

template <typename T>
[[nodiscard]] std::int32_t reduction_kind_for_()  // NOLINT(readability-identifier-naming)
{
  static const std::int32_t id = next_reduction_kind_();
  return id;
}

template <typename Fun>
[[nodiscard]] GlobalRedopID get_reduction_id_()  // NOLINT(readability-identifier-naming)
{
  static const GlobalRedopID id = []() -> GlobalRedopID {
    const LocalRedopID new_id           = reduction_id_for_<Fun>();
    const observer_ptr<Runtime> runtime = Runtime::get_runtime();
    Library library = runtime->find_or_create_library("legate.stl", LEGATE_STL_RESOURCE_CONFIG);

    return library.register_reduction_operator<Fun>(new_id);
  }();
  return id;
}

template <typename ElementType, typename Fun>
[[nodiscard]] std::int32_t record_reduction_for_()  // NOLINT(readability-identifier-naming)
{
  static const auto kind = []() -> std::int32_t {
    const Type type{primitive_type(type_code_of_v<ElementType>)};
    const GlobalRedopID id      = get_reduction_id_<Fun>();
    const std::int32_t red_kind = reduction_kind_for_<Fun>();

    type.record_reduction_operator(red_kind, id);
    return red_kind;
  }();
  return kind;
}
/**
 * @endcond
 */

template <typename...>
class Reduction;

/**
 * @cond
 * This specialization is used as an implementation detail of the `launch_task`
 * function.
 */
template <>
class Reduction<> {};
/**
 * @endcond
 */

/**
 * @brief Class template for reduction operations.
 *
 * This class template represents a reduction operation for a Legate task.
 * It stores a logical store and the reduction function and provides
 * a callable operator to add it as a reduction and a scalar argument to an `AutoTask` object.
 *
 * @ingroup stl-utilities
 */
template <typename Store, typename Fun>
class Reduction<Store, Fun> {
 public:
  LogicalStore data{
    nullptr}; /**< The logical store on which the reduction operation is performed. */
  Fun fn{};   /**< The reduction function to be applied. */

  /**
   * @brief Function call operator for the reduction operation.
   *
   * This function adds the reduction operation to an `AutoTask`.
   * It finds or declares the partition for the logical store, records the reduction operation,
   * and adds the reduction function as a scalar arguments to the task.
   *
   * @param task The AutoTask on which the reduction operation is invoked.
   *
   * @ingroup stl-utilities
   */
  void operator()(AutoTask& task) const
  {
    auto part       = task.find_or_declare_partition(data);
    const auto kind = record_reduction_for_<element_type_of_t<Store>, Fun>();

    task.add_reduction(data, kind, std::move(part));
    task.add_scalar_arg(Scalar{binary_type(sizeof(fn)), std::addressof(fn), true});
  }
};

/**
 * @cond
 */
enum class StoreType : std::uint8_t { INPUT, OUTPUT, REDUCTION };

class StorePlaceholder {
 public:
  StoreType which{};
  int index{};

 private:
  template <typename, typename>
  friend class Align;

  [[nodiscard]] LogicalStore operator()(const std::vector<LogicalStore>& inputs,
                                        const std::vector<LogicalStore>& outputs,
                                        const LogicalStore& reduction) const
  {
    switch (which) {
      case StoreType::INPUT: return inputs[index];
      case StoreType::OUTPUT: return outputs[index];
      case StoreType::REDUCTION: return reduction;
    }
    // src/legate/experimental/stl/detail/launch_task.hpp:238:3: error: control reaches end
    // of non-void function [-Werror=return-type]
    //
    // ... I mean, it doesn't, since that switch above is fully covered...
    LEGATE_UNREACHABLE();
  }
};

template <typename Left, typename Right>
class Align {
 public:
  Align(Left left, Right right) : left_{std::move(left)}, right_{std::move(right)} {}

  void operator()(AutoTask& task,
                  const std::vector<LogicalStore>& inputs,
                  const std::vector<LogicalStore>& outputs,
                  const LogicalStore& reduction) const
  {
    do_align_(task, left_(inputs, outputs, reduction), right_(inputs, outputs, reduction));
  }

 private:
  static void do_align_(AutoTask& task, Variable left, Variable right)
  {
    if (left.impl() != right.impl()) {
      task.add_constraint(legate::align(left, right));
    }
  }

  static void do_align_(AutoTask& task, const LogicalStore& left, const LogicalStore& right)
  {
    do_align_(task, task.find_or_declare_partition(left), task.find_or_declare_partition(right));
  }

  static void do_align_(AutoTask& task,
                        const LogicalStore& left,
                        const std::vector<LogicalStore>& right)
  {
    auto left_part = task.find_or_declare_partition(left);
    for (auto&& store : right) {
      do_align_(task, left_part, task.find_or_declare_partition(store));
    }
  }

  static void do_align_(AutoTask& task,
                        const std::vector<LogicalStore>& left,
                        const LogicalStore& right)
  {
    auto right_part = task.find_or_declare_partition(right);
    for (auto&& store : left) {
      do_align_(task, task.find_or_declare_partition(store), right_part);
    }
  }

  Left left_{};
  Right right_{};
};

class MakeInputs {
 public:
  template <typename... Ts>                  //
    requires(logical_store_like<Ts> && ...)  //
  [[nodiscard]] Inputs<std::remove_reference_t<Ts>...> operator()(Ts&&... stores) const
  {
    return Inputs<std::remove_reference_t<Ts>...>{
      std::vector<LogicalStore>{get_logical_store(std::forward<Ts>(stores))...}};
  }

  [[nodiscard]] StorePlaceholder operator[](int index) const { return {StoreType::INPUT, index}; }

 private:
  template <typename, typename>
  friend class Align;

  [[nodiscard]] const std::vector<LogicalStore>& operator()(
    const std::vector<LogicalStore>& inputs,
    const std::vector<LogicalStore>& /*outputs*/,
    const LogicalStore& /*reduction*/) const
  {
    return inputs;  // NOLINT(bugprone-return-const-ref-from-parameter)
  }
};

class MakeOutputs {
 public:
  template <typename... Ts>                  //
    requires(logical_store_like<Ts> && ...)  //
  [[nodiscard]] Outputs<std::remove_reference_t<Ts>...> operator()(Ts&&... stores) const
  {
    return Outputs<std::remove_reference_t<Ts>...>{
      std::vector<LogicalStore>{get_logical_store(std::forward<Ts>(stores))...}};
  }

  [[nodiscard]] StorePlaceholder operator[](int index) const { return {StoreType::OUTPUT, index}; }

 private:
  template <typename, typename>
  friend class Align;

  [[nodiscard]] const std::vector<LogicalStore>& operator()(
    const std::vector<LogicalStore>& /*inputs*/,
    const std::vector<LogicalStore>& outputs,
    const LogicalStore& /*reduction*/) const
  {
    return outputs;  // NOLINT(bugprone-return-const-ref-from-parameter)
  }
};

class MakeScalars {
 public:
  template <typename... Ts>
  [[nodiscard]] Scalars<Ts...> operator()(Ts&&... scalars) const
  {
    static_assert((std::is_trivially_copyable_v<std::decay_t<Ts>> && ...),
                  "All scalar arguments must be trivially copyable");
    return {{std::forward<Ts>(scalars)...}};
  }
};

class MakeFunction {
 public:
  template <typename Fn>
  [[nodiscard]] Function<std::decay_t<Fn>> operator()(Fn&& fn) const
  {
    return {std::forward<Fn>(fn)};
  }
};

template <typename Store>
using dim_of_t = meta::constant<dim_of_v<Store>>;

class MakeReduction {
 public:
  template <typename Store, typename ReductionFn>  //
    requires(logical_store_like<Store>)            // TODO(ericniebler): constrain Fun
  [[nodiscard]] Reduction<std::remove_reference_t<Store>, std::decay_t<ReductionFn>> operator()(
    Store&& store, ReductionFn&& reduction) const
  {
    static_assert(legate_reduction<ReductionFn>,
                  "The stl::reduction() function requires a Legate reduction operation "
                  "such as legate::SumReduction or legate::MaxReduction");
    return {get_logical_store(std::forward<Store>(store)), std::forward<ReductionFn>(reduction)};
  }

 private:
  template <typename, typename>
  friend class Align;

  [[nodiscard]] LogicalStore operator()(const std::vector<LogicalStore>& /*inputs*/,
                                        const std::vector<LogicalStore>& /*outputs*/,
                                        const LogicalStore& reduction) const
  {
    return reduction;
  }
};

class MakeConstraints {
 public:
  template <typename... Ts>  //
    requires((callable<Ts,
                       AutoTask&,
                       const std::vector<LogicalStore>&,
                       const std::vector<LogicalStore>&,
                       const LogicalStore&> &&
              ...))
  [[nodiscard]] Constraints<std::decay_t<Ts>...> operator()(Ts&&... constraints) const
  {
    return {{std::forward<Ts>(constraints)...}};
  }
};

class MakeAlign {
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
  [[nodiscard]] Align<Left, Right> operator()(Left left, Right right) const
  {
    return {left, right};
  }

  // For `align(inputs)`
  [[nodiscard]] auto operator()(MakeInputs inputs) const { return (*this)(inputs[0], inputs); }

  // For `align(outputs)`
  [[nodiscard]] auto operator()(MakeOutputs outputs) const { return (*this)(outputs[0], outputs); }
};

namespace cpu_detail {

template <typename Fn, typename... Views>
void cpu_for_each(Fn fn, Views... views)  // NOLINT(performance-unnecessary-value-param)
{
  auto&& input0 = front_of(views...);
  auto&& begin  = input0.begin();

  static_assert_iterator_category<std::forward_iterator_tag>(begin);

  const auto distance = std::distance(std::move(begin), input0.end());

  for (std::int64_t idx = 0; idx < distance; ++idx) {  //
    fn(*(views.begin() + idx)...);
  }
}

}  // namespace cpu_detail

template <typename Function, typename inputs, typename Outputs, typename Scalars>
class IterationCPU;

// This is a CPU implementation of a for_each operation.
template <typename Fn, typename... Is, typename... Os, typename... Ss>
class IterationCPU<Function<Fn>, Inputs<Is...>, Outputs<Os...>, Scalars<Ss...>> {
 public:
  template <std::size_t... InputsIs, std::size_t... OutputIs, std::size_t... ScalarsIs>
  static void impl(std::index_sequence<InputsIs...>,
                   std::index_sequence<OutputIs...>,
                   std::index_sequence<ScalarsIs...>,
                   const std::vector<PhysicalArray>& inputs,
                   std::vector<PhysicalArray>& outputs,
                   const std::vector<Scalar>& scalars)
  {
    cpu_detail::cpu_for_each(
      stl::bind_back(scalar_cast<const Fn&>(scalars[0]),
                     scalar_cast<const Ss&>(scalars[ScalarsIs + 1])...),
      Is::policy::physical_view(
        as_mdspan<const stl::value_type_of_t<Is>, stl::dim_of_v<Is>>(inputs[InputsIs]))...,
      Os::policy::physical_view(
        as_mdspan<stl::value_type_of_t<Os>, stl::dim_of_v<Os>>(outputs[OutputIs]))...);
  }

  template <std::int32_t ActualDim>
  void operator()(const std::vector<PhysicalArray>& inputs,
                  std::vector<PhysicalArray>& outputs,
                  const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t DIM = dim_of_v<meta::front<Is...>>;

    if constexpr (DIM == ActualDim) {
      const Legion::Rect<DIM> shape = inputs[0].shape<DIM>();

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

#if LEGATE_DEFINED(LEGATE_USE_CUDA) && LEGATE_DEFINED(LEGATE_NVCC)

template <typename Fn, typename... Views>
LEGATE_KERNEL void _gpu_for_each(Fn fn, Views... views)
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

  template <std::size_t... InputIs, std::size_t... OutputIs, std::size_t... ScalarIs>
  static void impl(std::index_sequence<InputIs...>,
                   std::index_sequence<OutputIs...>,
                   std::index_sequence<ScalarIs...>,
                   const legate::TaskContext& context,
                   const std::vector<PhysicalArray>& inputs,
                   std::vector<PhysicalArray>& outputs,
                   const std::vector<Scalar>& scalars)
  {
    const auto stream            = context.get_task_stream();
    const std::size_t volume     = meta::front<Is...>::policy::size(inputs[0]);
    const std::size_t num_blocks = (volume + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    _gpu_for_each<<<num_blocks, THREAD_BLOCK_SIZE, 0, stream>>>(
      stl::bind_back(scalar_cast<const Fn&>(scalars[0]),
                     scalar_cast<const Ss&>(scalars[ScalarIs + 1])...),
      Is::policy::physical_view(
        as_mdspan<const stl::value_type_of_t<Is>, stl::dim_of_v<Is>>(inputs[InputIs]))...,
      Os::policy::physical_view(
        as_mdspan<stl::value_type_of_t<Os>, stl::dim_of_v<Os>>(outputs[OutputIs]))...);
  }

  template <std::int32_t ActualDim>
  void operator()(const legate::TaskContext& context,
                  const std::vector<PhysicalArray>& inputs,
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
             context,
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
struct IterationOperation  //
  : LegateTask<IterationOperation<Function, Inputs, Outputs, Scalars, Constraints>> {
  static constexpr auto CPU_VARIANT_OPTIONS = VariantOptions{}.with_has_allocations(false);
  static constexpr auto GPU_VARIANT_OPTIONS = CPU_VARIANT_OPTIONS;

  static void cpu_variant(TaskContext context)
  {
    auto&& inputs  = context.inputs();
    auto&& outputs = context.outputs();
    auto&& scalars = context.scalars();
    const auto dim = inputs.at(0).dim();

    dim_dispatch(dim, IterationCPU<Function, Inputs, Outputs, Scalars>{}, inputs, outputs, scalars);
  }

#if LEGATE_DEFINED(LEGATE_USE_CUDA) && LEGATE_DEFINED(LEGATE_NVCC)
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
      dim, iteration_gpu<Function, Inputs, Outputs, Scalars>{}, context, inputs, outputs, scalars);
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
  Mapping mapping{detail::dynamic_extents<Dim>(working_set)};  // NOLINT(misc-const-correctness)

  using Policy   = ReductionAccessor<Op, Exclusive>;
  using Accessor = detail::MDSpanAccessor<typename Op::RHS, Dim, Policy>;
  Accessor accessor{std::move(store), std::move(working_set)};  // NOLINT(misc-const-correctness)

  using Handle = typename Accessor::data_handle_type;
  Handle handle{};  // NOLINT(misc-const-correctness)

  return {std::move(handle), std::move(mapping), std::move(accessor)};
}

namespace cpu_detail {

template <typename Function, typename InputOutput, typename Input>
void cpu_reduce(Function&& fn, InputOutput&& input_output, Input&& input)
{
  // These need to be at least multi-pass
  static_assert_iterator_category<std::forward_iterator_tag>(input_output.begin());
  static_assert_iterator_category<std::forward_iterator_tag>(input.begin());
  const auto distance = std::distance(input_output.begin(), input_output.end());

  LEGATE_ASSERT(distance == std::distance(input.begin(), input.end()));
  for (std::int64_t idx = 0; idx < distance; ++idx) {
    fn(*(input_output.begin() + idx), *(input.begin() + idx));
  }
}

}  // namespace cpu_detail

template <typename Reduction, typename Inputs, typename Outputs, typename Scalars>
class ReductionCPU;

template <typename Red, typename Fn, typename... Is, typename... Os, typename... Ss>
class ReductionCPU<Reduction<Red, Fn>, Inputs<Is...>, Outputs<Os...>, Scalars<Ss...>> {
 public:
  template <std::size_t... InputIs, std::size_t... OutputIs, std::size_t... ScalarIs>
  static void impl(std::index_sequence<InputIs...>,
                   std::index_sequence<OutputIs...>,
                   std::index_sequence<ScalarIs...>,
                   PhysicalArray& reduction,
                   const std::vector<PhysicalArray>& inputs,
                   std::vector<PhysicalArray>& outputs,
                   const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t DIM = stl::dim_of_v<Red>;
    Rect<DIM> working_set      = reduction.shape<DIM>();
    ((working_set = working_set.intersection(inputs[InputIs].shape<DIM>())), ...);

    cpu_detail::cpu_reduce(
      stl::bind_back(scalar_cast<const Fn&>(scalars[0]),
                     scalar_cast<const Ss&>(scalars[ScalarIs + 1])...),
      Red::policy::physical_view(  //
        as_mdspan_reduction<Fn, DIM>(reduction, std::move(working_set))),
      Is::policy::physical_view(
        as_mdspan<const stl::value_type_of_t<Is>, stl::dim_of_v<Is>>(inputs[InputIs]))...,
      Os::policy::physical_view(
        as_mdspan<stl::value_type_of_t<Os>, stl::dim_of_v<Os>>(outputs[OutputIs]))...);
  }

  template <std::int32_t ActualDim>
  void operator()(std::vector<PhysicalArray>& reductions,
                  const std::vector<PhysicalArray>& inputs,
                  std::vector<PhysicalArray>& outputs,
                  const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t DIM = dim_of_v<Red>;

    if constexpr (DIM == ActualDim) {
      const Legion::Rect<DIM> shape = reductions.at(0).shape<DIM>();

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

#if LEGATE_DEFINED(LEGATE_USE_CUDA) && LEGATE_DEFINED(LEGATE_NVCC)

namespace gpu_detail {

// TODO: this can be parallelized as well with care to avoid data races.
// If the view types carried metadata about the stride that avoids interference,
// then we can launch several kernels, each of which folds in parallel at
// multiples of that stride, but starting at different offsets. Then those
// results can be folded together.
template <typename Function, typename InputOutput, typename Input>
LEGATE_KERNEL void gpu_reduce(Function fn, InputOutput input_output, Input input)
{
  const auto tid      = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto distance = input_output.end() - input_output.begin();

  LEGATE_ASSERT(distance == (input.end() - input.begin()));
  for (std::int64_t idx = 0; idx < distance; ++idx) {
    fn(tid, *(input_output.begin() + idx), *(input.begin() + idx));
  }
}

}  // namespace gpu_detail

template <typename Reduction, typename Inputs, typename Outputs, typename Scalars>
class reduction_gpu;

template <typename Red, typename Fn, typename... Is, typename... Os, typename... Ss>
class reduction_gpu<reduction<Red, Fn>, inputs<Is...>, outputs<Os...>, scalars<Ss...>> {
 public:
  static constexpr std::int32_t THREAD_BLOCK_SIZE = 128;

  template <std::size_t... InputIs, std::size_t... OutputIs, std::size_t... ScalarIs>
  static void impl(std::index_sequence<InputIs...>,
                   std::index_sequence<OutputIs...>,
                   std::index_sequence<ScalarIs...>,
                   const legate::TaskContext& context,
                   PhysicalArray& reduction,
                   const std::vector<PhysicalArray>& inputs,
                   std::vector<PhysicalArray>& outputs,
                   const std::vector<Scalar>& scalars)
  {
    const auto stream            = context.get_task_stream();
    const std::size_t volume     = Red::policy::size(reduction);
    const std::size_t num_blocks = (volume + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE;

    constexpr std::int32_t Dim = dim_of_v<Red>;
    Rect<Dim> working_set      = reduction.shape<Dim>();
    ((working_set = working_set.intersection(inputs[InputIs].shape<Dim>())), ...);

    gpu_detail::gpu_reduce<<<num_blocks, THREAD_BLOCK_SIZE, 0, stream>>>(
      stl::bind_back(scalar_cast<const Fn&>(scalars[0]),
                     scalar_cast<const Ss&>(scalars[ScalarIs + 1])...),
      Red::policy::physical_view(  //
        as_mdspan_reduction<Fn, Dim>(reduction, working_set)),
      Is::policy::physical_view(
        as_mdspan<const stl::value_type_of_t<Is>, stl::dim_of_v<Is>>(inputs[InputIs]))...,
      Os::policy::physical_view(
        as_mdspan<stl::value_type_of_t<Os>, stl::dim_of_v<Os>>(outputs[OutputIs]))...);
  }

  template <std::int32_t ActualDim>
  void operator()(const legate::TaskContext& context,
                  std::vector<PhysicalArray>& reductions,
                  const std::vector<PhysicalArray>& inputs,
                  std::vector<PhysicalArray>& outputs,
                  const std::vector<Scalar>& scalars)
  {
    constexpr std::int32_t DIM = dim_of_v<Red>;

    if constexpr (DIM == ActualDim) {
      const Legion::Rect<DIM> shape = reductions.at(0).shape<DIM>();

      if (!shape.empty()) {
        impl(std::index_sequence_for<Is...>{},
             std::index_sequence_for<Os...>{},
             std::index_sequence_for<Ss...>{},
             context,
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
struct ReductionOperation
  : LegateTask<ReductionOperation<Reduction, Inputs, Outputs, Scalars, Constraints>> {
  static constexpr auto CPU_VARIANT_OPTIONS = VariantOptions{}.with_has_allocations(false);
  static constexpr auto GPU_VARIANT_OPTIONS = CPU_VARIANT_OPTIONS;

  static void cpu_variant(TaskContext context)
  {
    auto&& inputs     = context.inputs();
    auto&& outputs    = context.outputs();
    auto&& scalars    = context.scalars();
    auto&& reductions = context.reductions();
    const auto dim    = reductions.at(0).dim();

    dim_dispatch(dim,
                 ReductionCPU<Reduction, Inputs, Outputs, Scalars>{},
                 reductions,
                 inputs,
                 outputs,
                 scalars);
  }

#if LEGATE_DEFINED(LEGATE_USE_CUDA) && LEGATE_DEFINED(LEGATE_NVCC)
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
                 reduction_gpu<Reduction, Inputs, Outputs, Scalars>{},
                 context,
                 reductions,
                 inputs,
                 outputs,
                 scalars);
  }
#endif
};

namespace gcc_9_detail {

// _is_which is needed to disambiguate between the two overloads of
// get_arg below for gcc-9.
template <template <typename...> typename Which, typename What>
inline constexpr bool is_which_v = false;

template <template <typename...> typename Which, typename... Args>
inline constexpr bool is_which_v<Which, Which<Args...>> = true;

}  // namespace gcc_9_detail

template <template <typename...> typename Which, typename... Ts, typename... Tail>
[[nodiscard]] inline const Which<Ts...>& get_arg(const Which<Ts...>& head, const Tail&...)
{
  return head;  // NOLINT(bugprone-return-const-ref-from-parameter)
}

template <template <typename...> typename Which,
          typename Head,
          std::enable_if_t<!gcc_9_detail::is_which_v<Which, Head>, int> Enable = 0,
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
 *
 * @ingroup stl-utilities
 */
class LaunchTask {
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
   *
   * @ingroup stl-utilities
   */
  template <typename Function,
            typename Inputs,
            typename Outputs,
            typename Scalars,
            typename Constraints>
  static void make_iteration_task_(Function&& function,
                                   Inputs&& inputs,
                                   Outputs&& outputs,
                                   Scalars&& scalars,
                                   Constraints&& constraints)
  {
    auto&& [task, runtime] = make_task_<IterationOperation<std::decay_t<Function>,
                                                           std::decay_t<Inputs>,
                                                           std::decay_t<Outputs>,
                                                           std::decay_t<Scalars>,
                                                           std::decay_t<Constraints>>>();

    inputs(task, iteration_kind{});
    outputs(task, iteration_kind{});
    function(task);  // must precede scalars
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
   *
   * @ingroup stl-utilities
   */
  template <typename Reduction,
            typename Inputs,
            typename Outputs,
            typename Scalars,
            typename Constraints>
  static void make_reduction_task_(Reduction&& reduction,
                                   Inputs&& inputs,
                                   Outputs&& outputs,
                                   Scalars&& scalars,
                                   Constraints&& constraints)
  {
    auto&& [task, runtime] = make_task_<ReductionOperation<std::decay_t<Reduction>,
                                                           std::decay_t<Inputs>,
                                                           std::decay_t<Outputs>,
                                                           std::decay_t<Scalars>,
                                                           std::decay_t<Constraints>>>();

    inputs(task, reduction_kind{});
    outputs(task, reduction_kind{});
    reduction(task);  // must precede scalars
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
   * @pre Either a `function<>` or a `reduction<>` argument must be specified.
   *
   * @ingroup stl-utilities
   */
  template <typename... Ts>
  void operator()(Ts&&... args) const
  {
    // TODO(ericniebler) these could also be made constexpr if we defined the inputs<> template
    // directly.
    const detail::Inputs<> no_inputs;
    const detail::Outputs<> no_outputs;
    constexpr detail::Scalars<> no_scalars;
    constexpr detail::Function<> no_function;
    constexpr detail::Reduction<> no_reduction;
    constexpr detail::Constraints<> no_constraints;

    auto function  = detail::get_arg<detail::Function>(args..., no_function);
    auto reduction = detail::get_arg<detail::Reduction>(args..., no_reduction);

    constexpr bool has_function  = !std::is_same_v<decltype(function), detail::Function<>>;
    constexpr bool has_reduction = !std::is_same_v<decltype(reduction), detail::Reduction<>>;

    if constexpr (has_function) {
      make_iteration_task_(std::move(function),
                           detail::get_arg<detail::Inputs>(args..., no_inputs),
                           detail::get_arg<detail::Outputs>(args..., no_outputs),
                           detail::get_arg<detail::Scalars>(args..., no_scalars),
                           detail::get_arg<detail::Constraints>(args..., no_constraints));
    } else if constexpr (has_reduction) {
      make_reduction_task_(std::move(reduction),
                           detail::get_arg<detail::Inputs>(args..., no_inputs),
                           detail::get_arg<detail::Outputs>(args..., no_outputs),
                           detail::get_arg<detail::Scalars>(args..., no_scalars),
                           detail::get_arg<detail::Constraints>(args..., no_constraints));
    } else {
      static_assert(has_function || has_reduction,
                    "You must specify either a function or a reduction");
    }
  }
};

}  // namespace detail

// NOLINTBEGIN(readability-identifier-naming)
inline constexpr detail::MakeInputs inputs{};
inline constexpr detail::MakeOutputs outputs{};
inline constexpr detail::MakeScalars scalars{};
inline constexpr detail::MakeFunction function{};
inline constexpr detail::MakeConstraints constraints{};
inline constexpr detail::MakeReduction reduction{};

inline constexpr detail::MakeAlign align{};
// NOLINTEND(readability-identifier-naming)
// TODO(ericniebler): broadcasting

/**
 * @cond
 */
inline constexpr detail::LaunchTask launch_task{};  // NOLINT(readability-identifier-naming)
/**
 * @endcond
 */

#if LEGATE_DEFINED(LEGATE_DOXYGEN)
/**
 * @brief A function that launches a task with the given inputs, outputs,
 * scalars, and constraints.
 *
 * Launch parameter arguments can be one of the following in any order:
 *
 * - `legate::experimental::stl::inputs` - specifies the input stores for the task
 *
 *    - \a Example:
 *
 *      @code
 *      inputs(store1, store2, store3)
 *      @endcode
 *
 * - `legate::experimental::stl::outputs` - specifies the output stores for the task
 *
 *    - \a Example:
 *
 *      @code
 *      outputs(store1, store2, store3)
 *      @endcode
 *
 * - `legate::experimental::stl::scalars` - specifies the scalar arguments for the task
 *
 *    - \a Example:
 *
 *      @code
 *      scalars(42, 3.14f)
 *      @endcode
 *
 * - `legate::experimental::stl::function` - specifies the function to be applied
 *    iteratively to the inputs.
 *
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
 *      @code{.cpp}
 *      function([](const auto& in, auto& out) { out = in * in; })
 *      @endcode
 *
 * - `legate::experimental::stl::reduction` - specifies the reduction store and
 *    the reduction function to be applied to the inputs.
 *
 *    - The function must be bitwise copyable.
 *    - The reduction function must take as `mdspan`s referring to parts
 *      of the input stores.
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
 *      @code{.cpp}
 *      stl::reduction(stl::rows_of(store), stl::elementwise(std::plus{}))
 *      @endcode
 *
 * - `legate::experimental::stl::constraints` - specifies the constraints for
 *    the task.
 *
 *    - A constraint is a callable that takes an `legate::AutoTask&` and
 *      the input, output, and reduction stores as arguments. Its function
 *      signature must be:
 *
 *      @code{.cpp}
 *      void(legate::AutoTask&,                // the task to add the constraints to
 *           const std::vector<LogicalStore>&, // the input stores
 *           const std::vector<LogicalStore>&, // the output stores
 *           const LogicalStore&)              // the reduction store
 *      @endcode
 *
 *    - Legate.STL provides one constraint generator,
 *      `legate::experimental::stl::align`, for specifying the alignment
 *      constraints for the task. It can be used many different ways:
 *
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
 * The following use of @c launch_task is equivalent to
 * <tt>stl::transform(input, output op)</tt>:
 * @snippet{trimleft} legate/experimental/stl/detail/transform.hpp stl-launch-task-doxygen-snippet
 *
 * @ingroup stl-utilities
 */
template <LaunchParam... Params>
void launch_task(Params... params);
#endif

}  // namespace legate::experimental::stl

#include <legate/experimental/stl/detail/suffix.hpp>
