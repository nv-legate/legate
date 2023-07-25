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

// Useful for IDEs
#include "core/runtime/runtime.h"

namespace legate {

namespace detail {

template <Core::RegistrationCallback CALLBACK>
void invoke_legate_registration_callback(Legion::Machine,
                                         Legion::Runtime*,
                                         const std::set<Processor>&)
{
  CALLBACK();
};

}  // namespace detail

template <Core::RegistrationCallback CALLBACK>
/*static*/ void Core::perform_registration()
{
  perform_callback(detail::invoke_legate_registration_callback<CALLBACK>);
}

}  // namespace legate
