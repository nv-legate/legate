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

#ifndef __LEGATE_IO_C_H__
#define __LEGATE_IO_C_H__

enum LegateIOOpCode {
  _OP_CODE_BASE = 0,
  READ_EVEN_TILES,
  READ_FILE,
  READ_UNEVEN_TILES,
  WRITE_EVEN_TILES,
  WRITE_FILE,
  WRITE_UNEVEN_TILES,
};

#endif  // __LEGATE_IO_C_H__
