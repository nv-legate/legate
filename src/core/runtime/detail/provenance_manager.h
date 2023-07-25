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

#include <string>
#include <vector>

namespace legate::detail {

class ProvenanceManager {
 public:
  ProvenanceManager();

 public:
  const std::string& get_provenance();

  void set_provenance(const std::string& p);

  void reset_provenance();

  void push_provenance(const std::string& p);

  void pop_provenance();

  void clear_all();

 private:
  std::vector<std::string> provenance_;
};

}  // namespace legate::detail
