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

#include "core/data/detail/array.h"

namespace legate::detail {

inline std::vector<std::shared_ptr<Store>> Array::stores() const
{
  std::vector<std::shared_ptr<Store>> result;
  _stores(result);
  return result;
}

// ==========================================================================================

inline BaseArray::BaseArray(std::shared_ptr<Store> data, std::shared_ptr<Store> null_mask)
  : data_{std::move(data)}, null_mask_{std::move(null_mask)}
{
}

inline int32_t BaseArray::dim() const { return data_->dim(); }

inline ArrayKind BaseArray::kind() const { return ArrayKind::BASE; }

inline std::shared_ptr<Type> BaseArray::type() const { return data_->type(); }

inline bool BaseArray::nullable() const { return null_mask_ != nullptr; }

inline bool BaseArray::nested() const { return false; }

inline bool BaseArray::valid() const { return data_->valid(); }

inline std::shared_ptr<Store> BaseArray::data() const { return data_; }

inline Domain BaseArray::domain() const { return data_->domain(); }

// ==========================================================================================

inline ListArray::ListArray(std::shared_ptr<Type> type,
                            std::shared_ptr<BaseArray> descriptor,
                            std::shared_ptr<Array> vardata)
  : type_{std::move(type)}, descriptor_{std::move(descriptor)}, vardata_{std::move(vardata)}
{
}

inline int32_t ListArray::dim() const { return descriptor_->dim(); }

inline ArrayKind ListArray::kind() const { return ArrayKind::LIST; }

inline std::shared_ptr<Type> ListArray::type() const { return type_; }

inline bool ListArray::unbound() const { return descriptor_->unbound() || vardata_->unbound(); }

inline bool ListArray::nullable() const { return vardata_->nullable(); }

inline bool ListArray::nested() const { return true; }

inline std::shared_ptr<Store> ListArray::null_mask() const { return descriptor_->null_mask(); }

inline std::shared_ptr<Array> ListArray::descriptor() const { return descriptor_; }

inline std::shared_ptr<Array> ListArray::vardata() const { return vardata_; }

inline Domain ListArray::domain() const { return descriptor_->domain(); }

// ==========================================================================================

inline StructArray::StructArray(std::shared_ptr<Type> type,
                                std::shared_ptr<Store> null_mask,
                                std::vector<std::shared_ptr<Array>>&& fields)
  : type_{std::move(type)}, null_mask_{std::move(null_mask)}, fields_{std::move(fields)}
{
}

inline int32_t StructArray::dim() const { return fields_.front()->dim(); }

inline ArrayKind StructArray::kind() const { return ArrayKind::STRUCT; }

inline std::shared_ptr<Type> StructArray::type() const { return type_; }

inline bool StructArray::nullable() const { return null_mask_ != nullptr; }

inline bool StructArray::nested() const { return true; }

}  // namespace legate::detail
