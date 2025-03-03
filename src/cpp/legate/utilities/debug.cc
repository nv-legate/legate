/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/utilities/debug.h>

#include <legate/data/physical_store.h>
#include <legate/type/type_traits.h>
#include <legate/utilities/dispatch.h>

namespace legate {

namespace {  // anonymous

class PrintDenseArrayFn {
 public:
  template <Type::Code CODE, int DIM>
  [[nodiscard]] std::string operator()(const PhysicalStore& store) const
  {
    using T              = type_of_t<CODE>;
    const Rect<DIM> rect = store.shape<DIM>();
    return print_dense_array(store.read_accessor<T>(rect), rect);
  }
};

}  // namespace

// TODO(mpapadakis): Disabled while we find a workaround for operator<< missing for
// cuda::std::complex, see legate.internal#475
// std::string print_dense_array(const PhysicalStore& store)
// {
//   LEGATE_CHECK(store.is_readable());
//   return double_dispatch(store.dim(), store.code(), print_dense_array_fn{}, store);
// }

}  // namespace legate
