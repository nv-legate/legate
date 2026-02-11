/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <type_traits>
#include <utilities/utilities.h>

namespace test_task_config_basic {

class TaskConfigUnit : public DefaultFixture {};

static_assert(!std::is_default_constructible_v<legate::TaskConfig>);

TEST_F(TaskConfigUnit, Empty)
{
  constexpr auto task_id = legate::LocalTaskID{1};
  const auto config      = legate::TaskConfig{task_id};

  ASSERT_EQ(config.task_id(), task_id);
  ASSERT_EQ(config.task_signature(), std::nullopt);
  ASSERT_EQ(config.variant_options(), std::nullopt);
  ASSERT_EQ(config.store_mappings(), std::nullopt);
}

TEST_F(TaskConfigUnit, Signature)
{
  constexpr auto task_id = legate::LocalTaskID{2};
  const auto signature   = legate::TaskSignature{}.inputs(/*lower_bound=*/1, /*upper_bound=*/5);
  const auto config      = legate::TaskConfig{task_id}.with_signature(signature);

  ASSERT_EQ(config.task_id(), task_id);
  ASSERT_THAT(config.task_signature(), ::testing::Optional(signature));
  ASSERT_EQ(config.variant_options(), std::nullopt);
  ASSERT_EQ(config.store_mappings(), std::nullopt);
}

TEST_F(TaskConfigUnit, VariantOptions)
{
  constexpr auto task_id = legate::LocalTaskID{3};
  constexpr auto options =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(true);
  const auto config = legate::TaskConfig{task_id}.with_variant_options(options);

  ASSERT_EQ(config.task_id(), task_id);
  ASSERT_EQ(config.task_signature(), std::nullopt);

  // Need to do things this way instead of using
  //
  // ASSERT_THAT(config.variant_options(), ::testing::Optional(std::cref(options)));
  //
  // Because std::reference_wrapper didn't gain operator== until C++26!
  const auto& optional_vopts = config.variant_options();

  ASSERT_TRUE(optional_vopts.has_value());

  const auto& vopts = optional_vopts->get();  // NOLINT(bugprone-unchecked-optional-access)

  ASSERT_EQ(vopts, options);
}

TEST_F(TaskConfigUnit, StoreMappings)
{
  // Need to use std::vector here. Tried to use std::array and regular array[], but then
  // Googletest tries to make a copy of the values. This results in compile errors because
  // ProxyStoreMapping is not default constructable.
  const std::vector<legate::mapping::ProxyStoreMapping> expected = {
    {legate::proxy::inputs, legate::mapping::StoreTarget::FBMEM},
    {legate::proxy::outputs,
     legate::mapping::ProxyInstanceMappingPolicy{}.with_exact(true).with_redundant(false)}};
  const auto config   = legate::TaskConfig{legate::LocalTaskID{2}}.with_store_mappings({expected});
  const auto mappings = config.store_mappings();

  ASSERT_THAT(mappings, ::testing::Optional(::testing::ContainerEq(expected)));
}

TEST_F(TaskConfigUnit, EqSelf)
{
  constexpr auto task_id = legate::LocalTaskID{3};
  constexpr auto options =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(true);
  const auto signature =
    legate::TaskSignature{}
      .inputs(/*lower_bound=*/1, /*upper_bound=*/5)
      .constraints(
        {{legate::align(legate::proxy::inputs),
          legate::scale({1, 2, 3, 4}, legate::proxy::outputs[1], legate::proxy::inputs[1]),
          legate::broadcast(legate::proxy::inputs)}});
  const auto lhs_config =
    legate::TaskConfig{task_id}.with_variant_options(options).with_signature(signature);

  ASSERT_EQ(lhs_config, lhs_config);
}

TEST_F(TaskConfigUnit, Eq)
{
  constexpr auto task_id = legate::LocalTaskID{3};
  constexpr auto options =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(true);
  const auto signature =
    legate::TaskSignature{}
      .inputs(/*lower_bound=*/1, /*upper_bound=*/5)
      .constraints(
        {{legate::align(legate::proxy::inputs),
          legate::scale({1, 2, 3, 4}, legate::proxy::outputs[1], legate::proxy::inputs[1]),
          legate::broadcast(legate::proxy::inputs)}});

  const auto lhs_config =
    legate::TaskConfig{task_id}.with_variant_options(options).with_signature(signature);
  const auto rhs_config =
    legate::TaskConfig{task_id}.with_variant_options(options).with_signature(signature);

  ASSERT_EQ(lhs_config, rhs_config);
}

TEST_F(TaskConfigUnit, NotEq)
{
  constexpr auto task_id = legate::LocalTaskID{3};
  constexpr auto options =
    legate::VariantOptions{}.with_has_allocations(true).with_concurrent(true);
  const auto signature =
    legate::TaskSignature{}
      .inputs(/*lower_bound=*/1, /*upper_bound=*/5)
      .constraints(
        {{legate::align(legate::proxy::inputs),
          legate::scale({1, 2, 3, 4}, legate::proxy::outputs[1], legate::proxy::inputs[1]),
          legate::broadcast(legate::proxy::inputs)}});
  const auto lhs_config =
    legate::TaskConfig{task_id}.with_variant_options(options).with_signature(signature);

  const auto rhs_signature = legate::TaskSignature{}
                               .inputs(/*lower_bound=*/1, /*upper_bound=*/5)
                               .constraints({{legate::align(legate::proxy::inputs),
                                              legate::broadcast(legate::proxy::inputs)}});
  const auto rhs_config =
    legate::TaskConfig{task_id}.with_variant_options(options).with_signature(rhs_signature);

  ASSERT_NE(lhs_config, rhs_config);
}

}  // namespace test_task_config_basic
