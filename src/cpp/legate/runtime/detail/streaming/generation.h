/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace legate::detail {

class BufferBuilder;

/**
 * @brief A class to hold information about a particular streaming run.
 */
class StreamingGeneration {
 public:
  /**
   * @brief A unique ID that identifies a particular set of tasks all belonging to the same
   * generation.
   */
  std::uint32_t generation{};

  /**
   * @brief The number of tasks (logical tasks, not leaf tasks) belonging to a particular
   * generation.
   */
  std::uint32_t size{};

  /**
   * @brief Serialize this StreamingGeneration.
   *
   * @param buffer The buffer to serialize into.
   */
  void pack(BufferBuilder& buffer) const;
};

}  // namespace legate::detail
