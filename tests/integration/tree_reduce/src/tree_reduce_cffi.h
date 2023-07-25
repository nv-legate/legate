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

#ifndef __REGISTER_C__
#define __REGISTER_C__

#ifdef __cplusplus
extern "C" {
#endif

enum Constants {
  NUM_NORMAL_PRODUCER = 3,
  TILE_SIZE           = 10,
};

enum TreeReduceOpCode {
  PRODUCE_NORMAL  = 0,
  REDUCE_NORMAL   = 1,
  PRODUCE_UNBOUND = 2,
  REDUCE_UNBOUND  = 3,

};

void perform_registration(void);

#ifdef __cplusplus
}
#endif

#endif  // __REGISTER_C__
