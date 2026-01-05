/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/runtime/detail/argument_parsing/flags/fbmem.h>

#include <legate/cuda/detail/cuda_driver_api.h>
#include <legate/runtime/detail/argument_parsing/argument.h>
#include <legate/runtime/detail/argument_parsing/exceptions.h>
#include <legate/utilities/detail/traced_exception.h>

#include <realm/module_config.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>

namespace legate::detail {

void configure_fbmem(bool auto_config,
                     const Realm::ModuleConfig* cuda,
                     const Argument<std::int32_t>& gpus,
                     Argument<Scaled<std::int64_t>>* fbmem)
{
  auto& fbmem_value = fbmem->value_mut().unscaled_value_mut();

  // We should have auto-configured --gpus already at this point
  LEGATE_CHECK(gpus.value() >= 0);
  if (fbmem_value >= 0) {
    return;
  }

  if (gpus.value() == 0) {
    fbmem_value = 0;
    return;
  }

  if (auto_config && cuda) {
    std::size_t res_fbmem = 0;

    // Currently, we assume all GPUs are identical, so we can just use the first one
    try {
      const auto ctx = legate::cuda::detail::AutoPrimaryContext{0};

      std::tie(res_fbmem, std::ignore) = cuda::detail::get_cuda_driver_api()->mem_get_info();
    } catch (const std::exception&) {
      throw TracedException<AutoConfigurationError>{
        "Unable to determine the available GPU memory."};
    }

    // We want to allocate 95% of the available memory. But mem_get_info() returns its value in
    // bytes, which we also need to convert by our scaling factor.
    constexpr double FBMEM_FRACTION = 0.95;
    const auto alloc_frac           = FBMEM_FRACTION * static_cast<double>(res_fbmem);
    const auto MB                   = static_cast<double>(fbmem->value().scale());

    fbmem_value = static_cast<std::int64_t>(std::floor(alloc_frac / MB));
  } else {
    constexpr auto MINIMAL_MEM = 256;

    fbmem_value = MINIMAL_MEM;
  }
}

}  // namespace legate::detail
