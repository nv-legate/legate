/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma once

#include <legate/task/task.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>

namespace legate::detail {

// A base class template for internal tasks using the Legion calling convention
template <typename T>
class LegionTask : private LegateTask<T> {  // NOLINT(bugprone-crtp-constructor-accessibility)
 public:
  using BASE = LegionTask<T>;

  using LegateTask<T>::register_variants;

 protected:
  using LegateTask<T>::task_name_;

 private:
  template <typename, template <typename...> typename, bool>
  friend class VariantHelper;

  template <typename U, LegionVariantImpl<U> variant_fn, VariantCode variant_kind>
  static void task_wrapper_(const void* args,
                            std::size_t arglen,
                            const void* userdata,
                            std::size_t userlen,
                            Legion::Processor p);
};

}  // namespace legate::detail

#include <legate/task/detail/legion_task.inl>
