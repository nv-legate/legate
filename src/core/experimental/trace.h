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

#include <cstdint>
#include <memory>

namespace legate::experimental {

class Trace {
 public:
  explicit Trace(std::uint32_t trace_id);
  ~Trace();

  Trace(const Trace&)            = delete;
  Trace& operator=(const Trace&) = delete;
  Trace(Trace&&)                 = delete;
  Trace& operator=(Trace&&)      = delete;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_{};
};

}  // namespace legate::experimental
