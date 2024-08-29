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

#include "core/utilities/detail/env.h"

#include "legate.h"

#include <gtest/gtest.h>

class DefaultFixture : public ::testing::Test {
 public:
  void SetUp() override { ::testing::Test::SetUp(); }
  void TearDown() override
  {
    // This fixture is used for unit tests and others that don't use the runtime, so we should check
    // if we can issue an execution fence
    if (!legate::has_started()) {
      return;
    }
    // We need to make sure that all tasks from the test case are done so that any test failures
    // within the tasks are attributed correctly
    legate::Runtime::get_runtime()->issue_execution_fence(true);
  }
};

template <typename Config>
class RegisterOnceFixture : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    auto runtime = legate::Runtime::get_runtime();
    auto created = false;
    auto library = runtime->find_or_create_library(
      Config::LIBRARY_NAME, legate::ResourceConfig{}, nullptr, {}, &created);
    if (!created) {
      return;
    }
    Config::registration_callback(library);
  }
};
