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

#include <cstdint>

namespace legate::detail {

class Config {
 public:
  // Note, if you change any of these values, or add new ones, you must update Config::reset_()
  // accordingly
  static inline bool show_progress_requested       = false;
  static inline bool use_empty_task                = false;
  static inline bool synchronize_stream_view       = false;
  static inline bool log_mapping_decisions         = false;
  static inline bool log_partitioning_decisions    = false;
  static inline bool has_socket_mem                = false;
  static inline std::uint64_t max_field_reuse_size = 0;
  static inline bool warmup_nccl                   = false;

  static void parse();

 private:
  static void reset_() noexcept;
};

}  // namespace legate::detail
