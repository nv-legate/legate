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

#include "legate_defines.h"

#include <stdexcept>
#include <utility>

namespace legate::detail {

static const std::string BOTTOM;

ProvenanceManager::ProvenanceManager() { push_provenance(BOTTOM); }

const std::string& ProvenanceManager::get_provenance() const
{
  if (LegateDefined(LEGATE_USE_DEBUG)) assert(!provenance_.empty());
  return provenance_.top();
}

void ProvenanceManager::set_provenance(std::string p)
{
  if (LegateDefined(LEGATE_USE_DEBUG)) assert(!provenance_.empty());
  std::swap(provenance_.top(), p);
}

void ProvenanceManager::reset_provenance()
{
  if (LegateDefined(LEGATE_USE_DEBUG)) assert(!provenance_.empty());
  provenance_.top() = BOTTOM;
}

bool ProvenanceManager::has_provenance() const { return get_provenance() != BOTTOM; }

void ProvenanceManager::push_provenance(std::string p) { provenance_.push(std::move(p)); }

void ProvenanceManager::pop_provenance()
{
  if (provenance_.size() <= 1) {
    throw std::underflow_error{"can't pop from an empty provenance stack"};
  }
  provenance_.pop();
}

void ProvenanceManager::clear_all()
{
  // why std::stack has no .clear(), is beyond our ability to comprehend
  provenance_ = {};
  push_provenance(BOTTOM);
}

}  // namespace legate::detail
