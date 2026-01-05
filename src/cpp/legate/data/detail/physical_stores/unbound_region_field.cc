/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
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
  : partitioned_{partitioned}, out_{out}, fid_{fid}
{
}

void UnboundRegionField::bind_empty_data(std::int32_t ndim)
{
  DomainPoint extents;

  extents.dim = ndim;
  for (std::int32_t dim = 0; dim < ndim; ++dim) {
    extents[dim] = 0;
  }

  auto empty_buffer = create_buffer<std::int8_t>(/*size=*/0);

  out_.return_data(extents, fid_, empty_buffer.get_instance(), false);
  bound_ = true;
}

}  // namespace legate::detail
