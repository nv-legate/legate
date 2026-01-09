/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace create_unbound_physical_store_test {

namespace {

constexpr std::uint32_t UNBOUND_STORE_EXTENTS = 9;

enum class UnboundStoreOpCode : std::uint8_t {
  BIND_EMPTY,
  BIND_CREATED_BUFFER,
  BIND_BUFFER,
  BIND_UNTYPED_BUFFER,
  INVALID_BINDING,
  INVALID_DIM,
};

class UnboundStoreBindFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::PhysicalStore& store, UnboundStoreOpCode code) const
  {
    using T = legate::type_of_t<CODE>;

    switch (code) {
      case UnboundStoreOpCode::BIND_EMPTY: {
        store.bind_empty_data();
        break;
      }
      case UnboundStoreOpCode::BIND_CREATED_BUFFER: {
        ASSERT_NO_THROW(static_cast<void>(
          store.create_output_buffer<T, DIM>(legate::Point<DIM>{UNBOUND_STORE_EXTENTS}, true)));
        break;
      }
      case UnboundStoreOpCode::BIND_BUFFER: {
        auto buffer = store.create_output_buffer<T, DIM>(legate::Point<DIM>{UNBOUND_STORE_EXTENTS});

        store.bind_data(buffer, legate::Point<DIM>::ONES());
        break;
      }
      case UnboundStoreOpCode::BIND_UNTYPED_BUFFER: {
        /// [Bind an untyped buffer to an unbound store]
        constexpr auto num_elements      = 9;
        const auto element_size_in_bytes = store.type().size();
        constexpr auto UNTYPEED_DATA_DIM = 1;
        auto buffer = legate::create_buffer<std::int8_t, UNTYPEED_DATA_DIM>(num_elements *
                                                                            element_size_in_bytes);

        store.bind_untyped_data(buffer, legate::Point<UNTYPEED_DATA_DIM>{num_elements});
        /// [Bind an untyped buffer to an unbound store]
        LEGATE_CHECK(num_elements == UNBOUND_STORE_EXTENTS);
        break;
      }
      case UnboundStoreOpCode::INVALID_BINDING: {
        auto buffer =
          store.create_output_buffer<T, DIM>(legate::Point<DIM>{UNBOUND_STORE_EXTENTS}, true);

        ASSERT_THROW(store.bind_data(buffer, legate::Point<DIM>::ONES()), std::invalid_argument);
        ASSERT_THROW(store.bind_empty_data(), std::invalid_argument);
        break;
      }
      case UnboundStoreOpCode::INVALID_DIM: {
        constexpr std::int32_t INVALID_DIM = (DIM % LEGATE_MAX_DIM) + 1;

        ASSERT_THROW(static_cast<void>(
                       store.create_output_buffer<T>(legate::Point<INVALID_DIM>::ONES(), true)),
                     std::invalid_argument);

        // bind to buffer
        store.bind_empty_data();
        break;
      }
    }
  }
};

class UnboundStoreCreateFn {
 public:
  template <legate::Type::Code CODE, std::int32_t DIM>
  void operator()(const legate::PhysicalStore& store) const
  {
    using T = legate::type_of_t<CODE>;

    ASSERT_FALSE(store.is_future());
    ASSERT_TRUE(store.is_unbound_store());
    ASSERT_EQ(store.dim(), DIM);
    ASSERT_TRUE(store.valid());
    ASSERT_EQ(store.type().code(), legate::type_code_of_v<T>);
    ASSERT_EQ(store.code(), legate::type_code_of_v<T>);
    ASSERT_FALSE(store.transformed());

    ASSERT_THROW(static_cast<void>(store.shape<DIM>()), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(store.domain()), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(store.get_inline_allocation()), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(store.target()), std::invalid_argument);

    // Specific APIs for future/bound store
    ASSERT_THROW(static_cast<void>(store.scalar<T>()), std::invalid_argument);
    ASSERT_THROW(static_cast<void>(store.read_accessor<T, DIM>()), std::invalid_argument);

    store.bind_empty_data();
  }
};

class UnboundStoreCreateTask : public legate::LegateTask<UnboundStoreCreateTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

class UnboundStoreBindTask : public legate::LegateTask<UnboundStoreBindTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{2}};

  static void cpu_variant(legate::TaskContext context);

  static constexpr auto CPU_VARIANT_OPTIONS = legate::VariantOptions{}.with_has_allocations(true);
};

/*static*/ void UnboundStoreCreateTask::cpu_variant(legate::TaskContext context)
{
  auto store = context.output(0).data();

  legate::double_dispatch(store.dim(), store.code(), UnboundStoreCreateFn{}, store);
}

/*static*/ void UnboundStoreBindTask::cpu_variant(legate::TaskContext context)
{
  auto store   = context.output(0).data();
  auto op_code = context.scalar(0).value<UnboundStoreOpCode>();

  legate::double_dispatch(store.dim(), store.code(), UnboundStoreBindFn{}, store, op_code);
}

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_create_unbound_physical_store";

  static void registration_callback(legate::Library library)
  {
    UnboundStoreCreateTask::register_variants(library);
    UnboundStoreBindTask::register_variants(library);
  }
};

class CreateUnboundPhysicalStoreUnit : public RegisterOnceFixture<Config> {};

class CreateUnboundStoreTest
  : public CreateUnboundPhysicalStoreUnit,
    public ::testing::WithParamInterface<std::tuple<legate::Type, std::uint32_t>> {};

// NOLINTBEGIN(readability-magic-numbers)

std::vector<std::tuple<legate::Type, std::uint32_t>> unbound_store_cases()
{
  std::vector<std::tuple<legate::Type, std::uint32_t>> cases = {
    {legate::uint32(), 1},
    {legate::bool_(), 2},
    {legate::float16(), 3},
    {legate::float32(), 4},
  };

#if LEGATE_MAX_DIM >= 5
  cases.emplace_back(legate::float64(), 5);
#endif
#if LEGATE_MAX_DIM >= 6
  cases.emplace_back(legate::complex64(), 6);
#endif
#if LEGATE_MAX_DIM >= 7
  cases.emplace_back(legate::complex128(), 7);
#endif

  return cases;
}

// NOLINTEND(readability-magic-numbers)

INSTANTIATE_TEST_SUITE_P(CreateUnboundPhysicalStoreUnit,
                         CreateUnboundStoreTest,
                         ::testing::ValuesIn(unbound_store_cases()));

void bind_unbound_store_by_task(UnboundStoreOpCode op_code,
                                const legate::Type& type,
                                std::uint32_t dim)
{
  auto runtime       = legate::Runtime::get_runtime();
  auto context       = runtime->find_library(Config::LIBRARY_NAME);
  auto logical_store = runtime->create_store(type, dim);
  auto task          = runtime->create_task(context, UnboundStoreBindTask::TASK_CONFIG.task_id());

  task.add_output(logical_store);
  task.add_scalar_arg(legate::Scalar{op_code});
  runtime->submit(std::move(task));

  // Turns out to be a bound store here
  auto store = logical_store.get_physical_store();

  ASSERT_FALSE(store.is_unbound_store());
  ASSERT_FALSE(logical_store.unbound());
}

}  // namespace

TEST_P(CreateUnboundStoreTest, UnboundStoreCreation)
{
  const auto [type, dim] = GetParam();
  auto runtime           = legate::Runtime::get_runtime();
  auto context           = runtime->find_library(Config::LIBRARY_NAME);
  auto logical_store     = runtime->create_store(type, dim);
  auto task = runtime->create_task(context, UnboundStoreCreateTask::TASK_CONFIG.task_id());

  task.add_output(logical_store);
  runtime->submit(std::move(task));
}

TEST_P(CreateUnboundStoreTest, BindEmpty)
{
  const auto [type, dim] = GetParam();

  bind_unbound_store_by_task(UnboundStoreOpCode::BIND_EMPTY, type, dim);
}

TEST_P(CreateUnboundStoreTest, BindCreateBuffer)
{
  const auto [type, dim] = GetParam();

  bind_unbound_store_by_task(UnboundStoreOpCode::BIND_CREATED_BUFFER, type, dim);
}

TEST_P(CreateUnboundStoreTest, BindBuffer)
{
  const auto [type, dim] = GetParam();

  bind_unbound_store_by_task(UnboundStoreOpCode::BIND_BUFFER, type, dim);
}

TEST_P(CreateUnboundStoreTest, BindUntypedBuffer)
{
  const auto [type, ignored] = GetParam();

  bind_unbound_store_by_task(UnboundStoreOpCode::BIND_UNTYPED_BUFFER, type, 1);
}

TEST_P(CreateUnboundStoreTest, InvalidBinding)
{
  const auto [type, dim] = GetParam();

  bind_unbound_store_by_task(UnboundStoreOpCode::INVALID_BINDING, type, dim);
}

TEST_P(CreateUnboundStoreTest, InvalidDim)
{
  const auto [type, dim] = GetParam();

  bind_unbound_store_by_task(UnboundStoreOpCode::INVALID_DIM, type, dim);
}

}  // namespace create_unbound_physical_store_test
