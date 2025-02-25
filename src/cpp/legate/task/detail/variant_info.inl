/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
