/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <legate/redop/redop.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace test_task_store {

// Enums
enum class StoreType : std::uint8_t {
  NORMAL_STORE  = 0,
  UNBOUND_STORE = 1,
  SCALAR_STORE  = 2,
};

enum class TaskDataMode : std::uint8_t {
  INPUT     = 0,
  OUTPUT    = 1,
  REDUCTION = 2,
};

// Constants
constexpr std::int32_t INT_VALUE1  = 123;
constexpr std::int32_t INT_VALUE2  = 20;
constexpr std::int32_t SIMPLE_TASK = 0;

// SimpleTask template class
template <std::int32_t DIM>
class SimpleTask : public legate::LegateTask<SimpleTask<DIM>> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{SIMPLE_TASK + DIM}};

  static void cpu_variant(legate::TaskContext context)
  {
    ASSERT_EQ(context.num_scalars(), 3);

    auto data_mode  = context.scalar(0).value<TaskDataMode>();
    auto store_type = context.scalar(1).value<StoreType>();
    auto value      = context.scalar(2).value<std::int32_t>();

    const legate::PhysicalStore store = [&] {
      switch (data_mode) {
        case TaskDataMode::INPUT: return context.input(0).data(); break;
        case TaskDataMode::OUTPUT: return context.output(0).data(); break;
        case TaskDataMode::REDUCTION: return context.reduction(0).data(); break;
      }
      LEGATE_ABORT("Invalid data mode");
    }();

    if (store_type == StoreType::UNBOUND_STORE) {
      store.bind_empty_data();
      if (data_mode == TaskDataMode::OUTPUT && context.output(0).nullable()) {
        context.output(0).null_mask().bind_empty_data();
      }
      return;
    }

    auto shape = store.shape<DIM>();
    if (shape.empty()) {
      return;
    }

    switch (data_mode) {
      case TaskDataMode::INPUT: {
        auto acc = store.read_accessor<std::int32_t, DIM>();
        for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
          EXPECT_EQ(acc[*it], value);
        }
      } break;
      case TaskDataMode::OUTPUT: {
        auto acc = store.write_accessor<std::int32_t, DIM>();
        for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
          acc[*it] = value;
        }
      } break;
      case TaskDataMode::REDUCTION: {
        auto acc = store.reduce_accessor<legate::SumReduction<std::int32_t>, true, DIM>();
        for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
          acc[*it].reduce(value);
        }
      } break;
    }
  }
};

// Config class
class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_task_store";

  static void registration_callback(legate::Library library)
  {
#define REGISTER_VARIANTS(__DIM__) SimpleTask<__DIM__>::register_variants(library);
    LEGION_FOREACH_N(REGISTER_VARIANTS);
#undef REGISTER_VARIANTS
  }
};

// Base test class
class TaskStoreTests : public RegisterOnceFixture<Config> {};

// Utility structs
struct VerifyOutputBody {
  template <std::int32_t DIM>
  void operator()(const legate::PhysicalStore& store, std::int32_t expected_value)
  {
    auto shape = store.shape<DIM>();
    if (shape.empty()) {
      return;
    }
    auto acc = store.read_accessor<std::int32_t, DIM>(shape);
    for (legate::PointInRectIterator<DIM> it{shape}; it.valid(); ++it) {
      EXPECT_EQ(acc[*it], expected_value);
    }
  }
};

}  // namespace test_task_store
