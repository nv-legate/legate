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

#include "library.h"
#include "registry_cffi.h"

namespace rg {

class NoVariantTask : public Task<NoVariantTask, NO_VARIANT> {};

}  // namespace rg

namespace  // unnamed
{

static void __attribute__((constructor)) register_tasks()
{
  rg::NoVariantTask::register_variants();
}

}  // namespace
