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

#include "core/utilities/debug.h"

#include "core/type/type_traits.h"
#include "core/utilities/dispatch.h"

namespace legate {

namespace {  // anonymous

struct print_dense_array_fn {
  template <Type::Code CODE, int DIM>
  std::string operator()(const Store& store)
  {
    using T        = legate_type_of<CODE>;
    Rect<DIM> rect = store.shape<DIM>();
    return print_dense_array(store.read_accessor<T>(rect), rect);
  }
};

}  // namespace

std::string print_dense_array(const Store& store)
{
  assert(store.is_readable());
  return double_dispatch(store.dim(), store.code(), print_dense_array_fn{}, store);
}

}  // namespace legate
