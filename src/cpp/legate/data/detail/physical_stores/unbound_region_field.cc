/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <legate/data/detail/physical_stores/unbound_region_field.h>

#include <legate_defines.h>

#include <legate/data/buffer.h>
#include <legate/utilities/abort.h>
#include <legate/utilities/machine.h>
#include <legate/utilities/macros.h>
#include <legate/utilities/typedefs.h>

#include <cstddef>
#include <cstdint>

namespace legate::detail {

UnboundRegionField::UnboundRegionField(const Legion::OutputRegion& out,
                                       Legion::FieldID fid,
                                       bool partitioned)
  : partitioned_{partitioned},
    num_elements_{sizeof(std::size_t),
                  find_memory_kind_for_executing_processor(),
                  nullptr /*init_value*/,
                  alignof(std::size_t)},
    out_{out},
    fid_{fid}
{
}

ReturnValue UnboundRegionField::pack_weight() const
{
  if (LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    if (!bound_) {
      LEGATE_ABORT(
        "Found an uninitialized unbound store. Please make sure you return buffers to all unbound "
        "stores in the task");
    }
  }
  return {num_elements_, sizeof(std::size_t), alignof(std::size_t)};
}

void UnboundRegionField::bind_empty_data(std::int32_t ndim)
{
  update_num_elements(0);

  DomainPoint extents;

  extents.dim = ndim;
  for (std::int32_t dim = 0; dim < ndim; ++dim) {
    extents[dim] = 0;
  }

  auto empty_buffer = create_buffer<std::int8_t>(/*size=*/0);

  out_.return_data(extents, fid_, empty_buffer.get_instance(), false);
  bound_ = true;
}

void UnboundRegionField::update_num_elements(std::size_t num_elements)
{
  const AccessorWO<std::size_t, 1> acc{num_elements_, sizeof(num_elements), false};

  acc[0] = num_elements;
}

}  // namespace legate::detail
