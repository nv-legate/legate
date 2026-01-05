/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

class DefaultFixture : public ::testing::Test {
 public:
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
  // clang-tidy thinks this won't get instantiated if it isn't called, but it is called. Not
  // really sure what should be done to "fix" this.
  void SetUp() override  // NOLINT(portability-template-virtual-member-function)
  {
    DefaultFixture::SetUp();
    auto* const runtime = legate::Runtime::get_runtime();
    auto created        = false;
    const auto library  = runtime->find_or_create_library(
      Config::LIBRARY_NAME, legate::ResourceConfig{}, /*mapper=*/nullptr, {}, &created);

    if (created) {
      Config::registration_callback(library);
    }
  }
};
