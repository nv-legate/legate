/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "core/experimental/stl/detail/registrar.hpp"

#include <gtest/gtest.h>
#include <optional>

class BaseFixture : public ::testing::Test {
 public:
  static void init(int argc, char** argv)
  {
    BaseFixture::argc_ = argc;
    BaseFixture::argv_ = argv;
  }

  static inline int argc_{};
  static inline char** argv_{};
};

class DefaultFixture : public BaseFixture {
 public:
  void SetUp() override
  {
    ASSERT_EQ(legate::start(argc_, argv_), 0);
    BaseFixture::SetUp();
  }

  void TearDown() override
  {
    ASSERT_EQ(legate::finish(), 0);
    BaseFixture::TearDown();
  }
};

class DeathTestFixture : public DefaultFixture {
 public:
  void SetUp() override
  {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    DefaultFixture::SetUp();
  }
};

using NoInitFixture = BaseFixture;

class DeathTestNoInitFixture : public NoInitFixture {
 public:
  void SetUp() override
  {
    GTEST_FLAG_SET(death_test_style, "threadsafe");
    NoInitFixture::SetUp();
  }
};

class LegateSTLFixture : public NoInitFixture {
 public:
  void SetUp() override
  {
    ASSERT_EQ(init_.emplace(argc_, argv_).result(), 0);
    NoInitFixture::SetUp();
  }

  void TearDown() override
  {
    init_.reset();
    NoInitFixture::TearDown();
  }

 private:
  inline static std::optional<legate::experimental::stl::initialize_library> init_{};
};
