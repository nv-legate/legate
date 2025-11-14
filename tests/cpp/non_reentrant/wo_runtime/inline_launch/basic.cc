/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <gtest/gtest.h>

#include <string_view>
#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_inline_launch_basic {

namespace {

class CheckTask : public legate::LegateTask<CheckTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(legate::TaskSignature{}.inputs(1));

  static void cpu_variant(legate::TaskContext context)
  {
    const auto arg = context.input(0).data();

    ASSERT_EQ(arg.target(), legate::mapping::StoreTarget::SYSMEM);
  }

  static void gpu_variant(legate::TaskContext context)
  {
    const auto arg = context.input(0).data();

    ASSERT_EQ(arg.target(), legate::mapping::StoreTarget::FBMEM);
  }

  static void omp_variant(legate::TaskContext context)
  {
    const auto arg = context.input(0).data();

    ASSERT_EQ(arg.target(), legate::mapping::StoreTarget::SOCKETMEM);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_inline_launch_basic";

  static void registration_callback(legate::Library library)
  {
    CheckTask::register_variants(library);
  }
};

class InlineLaunchUnit : public RegisterOnceFixture<Config> {
 public:
  void SetUp() override
  {
    ASSERT_NO_THROW(legate::start());
    RegisterOnceFixture::SetUp();
  }

  void TearDown() override
  {
    RegisterOnceFixture::TearDown();
    ASSERT_EQ(legate::finish(), 0);
  }

 private:
  legate::test::Environment::TemporaryEnvVar legate_config_{"LEGATE_CONFIG",
                                                            "--inline-task-launch ",
                                                            /* overwrite */ true};
};

[[nodiscard]] legate::LogicalStore make_store(const legate::Shape& shape)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto ret            = runtime->create_store(shape, legate::int64());

  runtime->issue_fill(ret, legate::Scalar{std::int64_t{0}});
  return ret;
}

class StoreTarget : public InlineLaunchUnit,
                    public ::testing::WithParamInterface<legate::mapping::TaskTarget> {};

}  // namespace

INSTANTIATE_TEST_SUITE_P(InlineLaunchUnit,
                         StoreTarget,
                         ::testing::Values(legate::mapping::TaskTarget::CPU,
                                           legate::mapping::TaskTarget::GPU,
                                           legate::mapping::TaskTarget::OMP));

TEST_P(StoreTarget, Basic)
{
  const auto target   = GetParam();
  auto* const runtime = legate::Runtime::get_runtime();
  const auto machine  = runtime->get_machine().only(target);

  if (machine.empty()) {
    GTEST_SKIP() << "Test requires " << target;
  }

  const auto _ = legate::Scope{machine};

  const auto lib   = runtime->find_library(Config::LIBRARY_NAME);
  auto task        = runtime->create_task(lib, CheckTask::TASK_CONFIG.task_id());
  const auto store = make_store(legate::Shape{5, 1});

  task.add_input(store);
  runtime->submit(std::move(task));
}

}  // namespace test_inline_launch_basic
