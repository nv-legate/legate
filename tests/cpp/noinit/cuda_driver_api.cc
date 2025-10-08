/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate/cuda/detail/cuda_driver_api.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <utilities/utilities.h>

namespace test_cuda_loader {

using CUDADriverAPITest = ::testing::Test;

TEST_F(CUDADriverAPITest, CreateDestroy)
{
  // The GCC 14 (and maybe others) warning is a false positive, so we suppress it locally.
  // We apply the pragma to the whole scope where the optional object lives and is used,
  // to ensure its destructor is also covered.
  //
  // Sample error log:
  // In destructor 'std::unique_ptr<_Tp, _Dp>::~unique_ptr() [with _Tp = void; _Dp = int
  // (*)(void*)]',
  //     inlined from 'legate::cuda::detail::CUDADriverAPI::~CUDADriverAPI()' at
  //     /tmp/conda-croot/legate/work/src/cpp/legate/cuda/detail/cuda_driver_api.h:36:7,
  // ...
  // /path/to//gcc/x86_64-conda-linux-gnu/14.3.0/include/c++/bits/unique_ptr.h:398:19:
  // error: '((void**)((char*)&driver +
  // offsetof(std::optional<legate::cuda::detail::CUDADriverAPI>,std::optional<
  // legate::cuda::detail::CUDADriverAPI>::<unnamed>.std::_Optional_base<legate::cuda::detail::CUDADriverAPI,
  // false, false>::_M_payload.std::_Optional_payload<legate::cuda::detail::CUDADriverAPI, false,
  // false, false>::<unnamed>.std::_Optional_payload<legate::cuda::detail::CUDADriverAPI, true,
  // false,
  // false>::<unnamed>.std::_Optional_payload_base<legate::cuda::detail::CUDADriverAPI>::_M_payload)))[5]'
  // may be used uninitialized [-Werror=maybe-uninitialized]
  //   398 |         if (__ptr != nullptr)
  //       |             ~~~~~~^~~~~~~~~~
  // /path/to/legate/work/tests/cpp/noinit/cuda_driver_api.cc: In member function 'virtual void
  // test_cuda_loader::CUDADriverAPITest_CreateDestroy_Test::TestBody()':
  // /path/to/legate/work/tests/cpp/noinit/cuda_driver_api.cc:23:54: note: 'driver' declared here
  //    23 |   std::optional<legate::cuda::detail::CUDADriverAPI> driver{};
  //       |                                                      ^~~~~~
  // In destructor 'std::unique_ptr<_Tp, _Dp>::~unique_ptr() [with _Tp = void; _Dp = int
  // (*)(void*)]',
  //     inlined from 'legate::cuda::detail::CUDADriverAPI::~CUDADriverAPI()' at
  //     /path/to/legate/work/src/cpp/legate/cuda/detail/cuda_driver_api.h:36:7,
  // ...
  // /path/to/gcc/x86_64-conda-linux-gnu/14.3.0/include/c++/bits/unique_ptr.h:398:19:
  // error: '((void**)((char*)&driver +
  // offsetof(std::optional<legate::cuda::detail::CUDADriverAPI>,std::optional<
  // legate::cuda::detail::CUDADriverAPI>::<unnamed>.std::_Optional_base<legate::cuda::detail::CUDADriverAPI,
  // false, false>::_M_payload.std::_Optional_payload<legate::cuda::detail::CUDADriverAPI, false,
  // false, false>::<unnamed>.std::_Optional_payload<legate::cuda::detail::CUDADriverAPI, true,
  // false,
  // false>::<unnamed>.std::_Optional_payload_base<legate::cuda::detail::CUDADriverAPI>::_M_payload)))[5]'
  // may be used uninitialized [-Werror=maybe-uninitialized]
  //   398 |         if (__ptr != nullptr)
  //       |             ~~~~~~^~~~~~~~~~
  // /path/to/legate/work/tests/cpp/noinit/cuda_driver_api.cc: In member function 'virtual void
  // test_cuda_loader::CUDADriverAPITest_CreateDestroy_Test::TestBody()':
  // /path/to/legate/work/tests/cpp/noinit/cuda_driver_api.cc:23:54: note: 'driver' declared here
  //    23 |   std::optional<legate::cuda::detail::CUDADriverAPI> driver{};
  //       |                                                      ^~~~~~
  LEGATE_PRAGMA_PUSH();
  LEGATE_PRAGMA_GCC_IGNORE("-Wmaybe-uninitialized");

  {
    std::optional<legate::cuda::detail::CUDADriverAPI> driver{};

    // Multiple create/destroy should work (even if CUDA isn't found)
    driver.emplace("foo");
    driver.reset();

    driver.emplace("bar");
    driver.reset();
  }  // <-- driver's destructor runs here, while the pragma is active

  LEGATE_PRAGMA_POP();
}

TEST_F(CUDADriverAPITest, SetLoadPath)
{
  const std::string fpath = "/this/file/does/not/exist.so";
  const legate::cuda::detail::CUDADriverAPI driver{fpath};

  ASSERT_EQ(driver.handle_path(), fpath);
  ASSERT_FALSE(driver.is_loaded());
  ASSERT_THROW(driver.init(), std::logic_error);
}

TEST_F(CUDADriverAPITest, TestLoad)
{
  const std::string fpath =
    LEGATE_SHARED_LIBRARY_PREFIX "legate_dummy_cuda_driver" LEGATE_SHARED_LIBRARY_SUFFIX;
  const legate::cuda::detail::CUDADriverAPI driver{fpath};

  ASSERT_THAT(driver.handle_path().as_string_view(), ::testing::EndsWith(fpath));
  ASSERT_TRUE(driver.is_loaded());
  ASSERT_NO_THROW(driver.init());
}

}  // namespace test_cuda_loader
