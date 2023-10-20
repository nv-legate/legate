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

#include "core/runtime/detail/provenance_manager.h"

#include "legion.h"

#include "legate_defines.h"

#include <assert.h>
#include <stdexcept>

namespace legate::detail {

static const std::string BOTTOM = "";

ProvenanceManager::ProvenanceManager() { provenance_.push_back(BOTTOM); }

const std::string& ProvenanceManager::get_provenance()
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(provenance_.size() > 0); }
  return provenance_.back();
}

void ProvenanceManager::set_provenance(const std::string& p)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(provenance_.size() > 0); }
  provenance_.back() = p;
}

void ProvenanceManager::reset_provenance()
{
  if (LegateDefined(LEGATE_USE_DEBUG)) { assert(provenance_.size() > 0); }
  provenance_.back() = BOTTOM;
}

bool ProvenanceManager::has_provenance() const { return provenance_.back() != BOTTOM; }

void ProvenanceManager::push_provenance(const std::string& p) { provenance_.push_back(p); }

void ProvenanceManager::pop_provenance()
{
  if (provenance_.size() <= 1) {
    throw std::underflow_error("can't pop from an empty provenance stack");
  }
  provenance_.pop_back();
}

void ProvenanceManager::clear_all()
{
  provenance_.clear();
  provenance_.push_back(BOTTOM);
}

}  // namespace legate::detail
