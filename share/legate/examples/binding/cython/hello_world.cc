/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <hello_world.h>

#include <iostream>

namespace hello_world {

void HelloWorld::cpu_variant(legate::TaskContext) { std::cout << "Hello World!\n"; }

}  // namespace hello_world
