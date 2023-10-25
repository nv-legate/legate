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

#include "legate.h"
#include "utilities/utilities.h"

#include <gtest/gtest.h>

namespace unit {

using DeathTestExample = DeathTestFixture;

void KillProcess(int argc, char** argv)
{
  (void)legate::start(0, NULL);
  abort();
}

TEST_F(DeathTestExample, Simple)
{
  auto value           = getenv("REALM_BACKTRACE");
  bool realm_backtrace = value != nullptr && atoi(value) != 0;

  if (realm_backtrace) {
    // We can't check that the subprocess dies with SIGABRT when we run with REALM_BACKTRACE=1,
    // because Realm's signal handler doesn't propagate the signal, instead it exits right away
    EXPECT_EXIT(KillProcess(argc_, argv_), ::testing::ExitedWithCode(1), "");
  } else {
    EXPECT_EXIT(KillProcess(argc_, argv_), ::testing::KilledBySignal(SIGABRT), "");
  }
}

}  // namespace unit
