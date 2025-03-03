/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <legate/task/detail/variant_info.h>

namespace legate::detail {

inline VariantInfo::VariantInfo(VariantImpl body_,
                                const Legion::CodeDescriptor& code_desc_,
                                VariantOptions options_,
                                std::optional<InternalSharedPtr<TaskSignature>> signature_)
  : body{body_}, code_desc{code_desc_}, options{options_}, signature{std::move(signature_)}
{
}

}  // namespace legate::detail
