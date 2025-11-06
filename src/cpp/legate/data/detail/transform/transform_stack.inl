/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/data/detail/transform/transform_stack.h>

#include <utility>

namespace legate::detail {

inline TransformStack::TransformStack(private_tag,
                                      std::unique_ptr<StoreTransform>&& transform,
                                      InternalSharedPtr<TransformStack> parent)
  : transform_{std::move(transform)},
    parent_{std::move(parent)},
    convertible_{transform_->is_convertible() && parent_->is_convertible()}
{
}

inline TransformStack::TransformStack(std::unique_ptr<StoreTransform>&& transform,
                                      const InternalSharedPtr<TransformStack>& parent)
  : TransformStack{private_tag{}, std::move(transform), parent}
{
}

inline TransformStack::TransformStack(std::unique_ptr<StoreTransform>&& transform,
                                      InternalSharedPtr<TransformStack>&& parent)
  : TransformStack{private_tag{}, std::move(transform), std::move(parent)}
{
}

inline bool TransformStack::is_convertible() const { return convertible_; }

inline bool TransformStack::identity() const { return nullptr == transform_; }

template <typename VISITOR, typename T>
auto TransformStack::convert_(VISITOR visitor, T&& input) const
{
  if (identity()) {
    return input;
  }
  if (parent_->identity()) {
    return visitor(transform_, std::forward<T>(input));
  }
  return visitor(transform_, visitor(parent_, std::forward<T>(input)));
}

template <typename VISITOR, typename T>
auto TransformStack::invert_(VISITOR visitor, T&& input) const
{
  if (identity()) {
    return input;
  }

  auto result = visitor(transform_, std::forward<T>(input));

  if (parent_->identity()) {
    return result;
  }
  return visitor(parent_, std::move(result));
}

}  // namespace legate::detail
