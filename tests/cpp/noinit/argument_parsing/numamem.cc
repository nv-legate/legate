/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/numamem.h>

#include <legate/runtime/detail/argument_parsing/argument.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utilities/utilities.h>
#include <vector>

namespace test_configure_numamem {

constexpr auto MB = 1 << 20;

class ConfigureNUMAMemUnit : public DefaultFixture, public ::testing::WithParamInterface<bool> {};

INSTANTIATE_TEST_SUITE_P(,
                         ConfigureNUMAMemUnit,
                         ::testing::Bool(),
                         ::testing::PrintToStringParamName{});

using ScaledType  = legate::detail::Scaled<std::int64_t>;
using NUMAMemType = legate::detail::Argument<ScaledType>;
using OpenMPsType = legate::detail::Argument<std::int32_t>;

TEST_P(ConfigureNUMAMemUnit, Preset)
{
  constexpr auto NUMA_SIZE = 128;
  const auto omps          = OpenMPsType{nullptr, "--omps", 1};
  auto numamem             = NUMAMemType{nullptr, "--numamems", ScaledType{NUMA_SIZE, MB, "MiB"}};

  legate::detail::configure_numamem(
    /* auto_config */ GetParam(), /* numa_mems */ {}, omps, &numamem);
  ASSERT_EQ(numamem.value().unscaled_value(), NUMA_SIZE);
}

TEST_P(ConfigureNUMAMemUnit, NoOpenMP)
{
  const auto omps = OpenMPsType{nullptr, "--omps", 0};
  auto numamem    = NUMAMemType{nullptr, "--numamems", ScaledType{-1, MB, "MiB"}};

  legate::detail::configure_numamem(
    /* auto_config */ GetParam(), /* numa_mems */ {}, omps, &numamem);
  ASSERT_EQ(numamem.value().unscaled_value(), 0);
}

TEST_P(ConfigureNUMAMemUnit, NoNUMAMem)
{
  const auto omps = OpenMPsType{nullptr, "--omps", 10};
  auto numamem    = NUMAMemType{nullptr, "--numamems", ScaledType{-1, MB, "MiB"}};

  legate::detail::configure_numamem(
    /* auto_config */ GetParam(), /* numa_mems */ {}, omps, &numamem);
  ASSERT_EQ(numamem.value().unscaled_value(), 0);
}

TEST_P(ConfigureNUMAMemUnit, NUMAMemNotDivisible)
{
  const auto omps      = OpenMPsType{nullptr, "--omps", 10};
  const auto numa_mems = std::vector<std::size_t>(static_cast<std::size_t>(omps.value() + 1), 0);
  auto numamem         = NUMAMemType{nullptr, "--numamems", ScaledType{-1, MB, "MiB"}};

  // NUMA mems aren't neatly divisible by the number of openmp threads
  ASSERT_NE(omps.value() % numa_mems.size(), 0);
  legate::detail::configure_numamem(
    /* auto_config */ GetParam(), numa_mems, omps, &numamem);
  ASSERT_EQ(numamem.value().unscaled_value(), 0);
}

TEST_F(ConfigureNUMAMemUnit, NoAutoConfig)
{
  constexpr auto MINIMAL_MEM = 256;
  const auto omps            = OpenMPsType{nullptr, "--omps", 10};
  const auto numa_mems       = std::vector<std::size_t>(static_cast<std::size_t>(omps.value()), 0);
  auto numamem               = NUMAMemType{nullptr, "--numamems", ScaledType{-1, MB, "MiB"}};

  legate::detail::configure_numamem(
    /* auto_config */ false, numa_mems, omps, &numamem);
  ASSERT_EQ(numamem.value().unscaled_value(), MINIMAL_MEM);
}

TEST_F(ConfigureNUMAMemUnit, AutoConfig)
{
  const auto omps      = OpenMPsType{nullptr, "--omps", 10};
  const auto numa_mems = std::vector<std::size_t>(static_cast<std::size_t>(omps.value()), 0);

  constexpr double SYSMEM_FRACTION = 0.8;
  const auto numa_mem_size         = numa_mems.front();
  const auto num_numa_mems         = numa_mems.size();
  const auto omps_per_numa         = (omps.value() + num_numa_mems - 1) / num_numa_mems;
  const auto auto_numamem =
    static_cast<std::int64_t>(std::floor(SYSMEM_FRACTION * static_cast<double>(numa_mem_size) / MB /
                                         static_cast<double>(omps_per_numa)));
  auto numamem = NUMAMemType{nullptr, "--numamems", ScaledType{-1, MB, "MiB"}};

  legate::detail::configure_numamem(
    /* auto_config */ true, numa_mems, omps, &numamem);
  ASSERT_EQ(numamem.value().scaled_value(), auto_numamem);
}

}  // namespace test_configure_numamem
