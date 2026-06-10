/*
 * SPDX-FileCopyrightText: Copyright (c) 2026-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate.h>

namespace legate::test {

class MockMapperRuntime : public Legion::Mapping::MapperRuntime {
 public:
  MockMapperRuntime() : MapperRuntime{nullptr} {}
};

}  // namespace legate::test
