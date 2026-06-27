/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda_runtime.h>
#include <repartition.h>

void InitTask::gpu_variant(legate::TaskContext context) { (void)context; };

void RepartitionTask::gpu_variant(legate::TaskContext context) { (void)context; }
