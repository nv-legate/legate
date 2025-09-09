/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/io/hdf5/interface.h>
#include <legate/utilities/detail/linearize.h>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <filesystem>
#include <limits>
#include <random>
#include <string>
#include <string_view>
#include <utilities/utilities.h>

namespace test_io_hdf5_write {

namespace {

[[nodiscard]] legate::DomainPoint to_point(legate::Span<const std::uint64_t> point)
{
  legate::DomainPoint ret{};

  LEGATE_CHECK(point.size() < std::numeric_limits<int>::max());
  ret.dim = static_cast<int>(point.size());
  std::copy_n(point.begin(), point.size(), ret.point_data);
  return ret;
}

// TODO(jfaibussowit)
// Remove this and merge https://github.com/nv-legate/legate.internal/pull/1604
template <std::int32_t DIM>
class TypeDispatcher : public legate::detail::InnerTypeDispatchFn<DIM> {
 public:
  using base = legate::detail::InnerTypeDispatchFn<DIM>;

  template <typename Functor, typename... Fnargs>
  constexpr decltype(auto) operator()(legate::Type::Code code, Functor&& f, Fnargs&&... args)
  {
    if (code == legate::Type::Code::BINARY) {
      return f.template operator()<legate::Type::Code::BINARY, DIM>(std::forward<Fnargs>(args)...);
    }
    return static_cast<base&>(*this)(code, std::forward<Functor>(f), std::forward<Fnargs>(args)...);
  }
};

template <typename F, typename... T>
void double_dispatch_with_binary(std::int32_t DIM,
                                 legate::Type::Code CODE,
                                 F&& functor,
                                 T&&... args)
{
  // The below is just an unrolled legate::double_dispatch() because double_dispatch() does
  // not support Type::Code::BINARY yet, and it won't until
  // https://github.com/nv-legate/legate.internal/pull/1604 is resolved/merged. This cludge
  // was done for the 25.01 release.

#define TYPE_DISPATCH(__dim__)                                                           \
  case __dim__: {                                                                        \
    TypeDispatcher<__dim__>{}(CODE, std::forward<F>(functor), std::forward<T>(args)...); \
    break;                                                                               \
  }

  switch (DIM) {
    LEGION_FOREACH_N(TYPE_DISPATCH);
    default: {  // legate-lint: no-switch-default
      legate::detail::throw_unsupported_dim(DIM);
    }
  }

#undef TYPE_DISPATCH
}

class IotaTask : public legate::LegateTask<IotaTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}}.with_signature(
      legate::TaskSignature{}.outputs(1).scalars(1));

  class Iota {
   public:
    template <legate::Type::Code CODE, std::int32_t DIM>
    void operator()(const legate::DomainPoint& lo,
                    const legate::DomainPoint& hi,
                    legate::PhysicalStore* store) const
    {
      using T =
        std::conditional_t<CODE == legate::Type::Code::BINARY, std::byte, legate::type_of_t<CODE>>;
      const auto acc =
        store->write_accessor<T, DIM, /* VALIDATE_TYPE */ CODE != legate::Type::Code::BINARY>();
      const auto shape = store->shape<DIM>();

      for (auto it = legate::PointInRectIterator<DIM>{shape}; it.valid(); ++it) {
        const auto lin_idx = legate::detail::linearize(lo, hi, *it);

        acc[*it] = static_cast<T>(lin_idx);
      }
    }
  };

  static void cpu_variant(legate::TaskContext context)
  {
    const auto global_shape = to_point(context.scalar(0).values<std::uint64_t>());
    const auto zero         = [&] {
      legate::DomainPoint pt{};

      pt.dim = global_shape.get_dim();
      return pt;
    }();
    auto store = context.output(0).data();

    double_dispatch_with_binary(store.dim(), store.code(), Iota{}, zero, global_shape, &store);
  }
};

class CheckerTask : public legate::LegateTask<CheckerTask> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{1}}.with_signature(
      legate::TaskSignature{}.inputs(2).constraints({{legate::align(legate::proxy::inputs)}}));

  class CheckImpl {
   public:
    template <legate::Type::Code CODE, std::int32_t DIM>
    void operator()(const legate::PhysicalArray& array,
                    const legate::PhysicalArray& array_copy) const
    {
      using T =
        std::conditional_t<CODE == legate::Type::Code::BINARY, std::byte, legate::type_of_t<CODE>>;
      const auto arr_acc =
        array.data()
          .span_read_accessor<T, DIM, /* VALIDATE_TYPE */ CODE != legate::Type::Code::BINARY>();
      const auto arr_cp_acc =
        array_copy.data()
          .span_read_accessor<T, DIM, /* VALIDATE_TYPE */ CODE != legate::Type::Code::BINARY>();

      ASSERT_EQ(arr_acc.extents(), arr_cp_acc.extents());
      legate::for_each_in_extent(arr_acc.extents(), [&](auto... idx) {
        ASSERT_EQ(arr_acc(idx...), arr_cp_acc(idx...))
          << "at index " << fmt::format("({})", fmt::join({idx...}, ", "));
      });
    }
  };

  static void cpu_variant(legate::TaskContext context)
  {
    const auto array      = context.input(0);
    const auto array_copy = context.input(1);

    ASSERT_EQ(array.dim(), array_copy.dim());
    ASSERT_EQ(array.data().type(), array_copy.data().type());

    double_dispatch_with_binary(array.dim(), array.data().code(), CheckImpl{}, array, array_copy);
  }
};

class Config {
 public:
  static constexpr std::string_view LIBRARY_NAME = "test_io_hdf5_write";

  static void registration_callback(legate::Library library)
  {
    IotaTask::register_variants(library);
    CheckerTask::register_variants(library);
  }
};

class IOHDF5WriteUnit : public RegisterOnceFixture<Config>,
                        public ::testing::WithParamInterface<legate::Type> {
 public:
  void TearDown() override { RegisterOnceFixture::TearDown(); }

  ~IOHDF5WriteUnit() override { static_cast<void>(std::filesystem::remove_all(base_path)); }

 private:
  [[nodiscard]] static std::string random_dir_name_()
  {
    static constexpr std::string_view chars{
      "_"
      "123456789"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "abcdefghijklmnopqrstuvwxyz"};
    constexpr auto LEN = 16;

    auto rng  = std::random_device{};
    auto dist = std::uniform_int_distribution{{}, chars.size() - 1};
    auto ret  = std::string{};

    std::generate_n(std::back_inserter(ret), LEN, [&] {
      char chr{};

      do {
        chr = chars[dist(rng)];
      } while (chr == '\0');
      return chr;
    });
    return fmt::format("legate_{}_{}", Config::LIBRARY_NAME, ret);
  }

 public:
  // If the test is run using multiple processes simultaneously, then each process should dump
  // its stuff to a separate directory, otherwise TearDown() will delete the directory out from
  // underneath a running process.
  // NOLINTNEXTLINE(cert-err58-cpp)
  static inline auto base_path = std::filesystem::temp_directory_path() / random_dir_name_();
};

[[nodiscard]] bool supported_by_hdf5_reader(const legate::Type& type)
{
  switch (type.code()) {
    case legate::Type::Code::BOOL: [[fallthrough]];
    case legate::Type::Code::UINT8: [[fallthrough]];
    case legate::Type::Code::UINT16: [[fallthrough]];
    case legate::Type::Code::UINT32: [[fallthrough]];
    case legate::Type::Code::UINT64: [[fallthrough]];
    case legate::Type::Code::INT8: [[fallthrough]];
    case legate::Type::Code::INT16: [[fallthrough]];
    case legate::Type::Code::INT32: [[fallthrough]];
    case legate::Type::Code::INT64: [[fallthrough]];
    case legate::Type::Code::FLOAT16: [[fallthrough]];
    case legate::Type::Code::FLOAT32: [[fallthrough]];
    case legate::Type::Code::FLOAT64: [[fallthrough]];
    case legate::Type::Code::NIL: [[fallthrough]];
    case legate::Type::Code::STRING: [[fallthrough]];
    case legate::Type::Code::BINARY: return true;
    // HDF5 does not have native complex support, we should approximate this with structure
    // types ideally.
    case legate::Type::Code::COMPLEX64: [[fallthrough]];
    case legate::Type::Code::COMPLEX128: [[fallthrough]];
    // These are not supported from our side
    case legate::Type::Code::FIXED_ARRAY: [[fallthrough]];
    case legate::Type::Code::STRUCT: [[fallthrough]];
    case legate::Type::Code::LIST: return false;
  }
  LEGATE_ABORT("Unhandled type code ", type.code());
}

}  // namespace

INSTANTIATE_TEST_SUITE_P(,
                         IOHDF5WriteUnit,
                         ::testing::Values(legate::bool_(),
                                           legate::int8(),
                                           legate::int16(),
                                           legate::int32(),
                                           legate::int64(),
                                           legate::uint8(),
                                           legate::uint16(),
                                           legate::uint32(),
                                           legate::uint64(),
                                           legate::float16(),
                                           legate::float32(),
                                           legate::float64(),
                                           legate::binary_type(10)));

TEST_P(IOHDF5WriteUnit, Basic)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto lib      = runtime->find_library(Config::LIBRARY_NAME);

  const auto& type          = GetParam();
  constexpr auto SHAPE_SIZE = 5;
  const auto shape          = legate::Shape{SHAPE_SIZE, SHAPE_SIZE, SHAPE_SIZE};
  const auto array          = runtime->create_array(shape, type);

  {
    auto task = runtime->create_task(lib, IotaTask::TASK_CONFIG.task_id());

    task.add_scalar_arg(legate::Scalar{shape.extents() - 1});
    task.add_output(array);
    runtime->submit(std::move(task));
  }

  const auto h5_file     = base_path / "foo.h5";
  constexpr auto dataset = "my_dataset";

  legate::io::hdf5::to_file(array, h5_file, dataset);

  if (!supported_by_hdf5_reader(type)) {
    GTEST_SUCCEED()
      << "Some types aren't properly supported by our reader, and/or are limited by "
         "bugs in HDF5. Return early for now, with the hope that things are eventually fixed.";
    return;
  }

  // Must block here so that the file is definitely on disk.
  runtime->issue_execution_fence(/* block */ true);

  const auto array_copy = legate::io::hdf5::from_file(h5_file, dataset);

  ASSERT_EQ(array_copy.dim(), array.dim());
  ASSERT_EQ(array_copy.shape(), array.shape());
  ASSERT_EQ(array_copy.type(), array.type());

  {
    auto task = runtime->create_task(lib, CheckerTask::TASK_CONFIG.task_id());

    task.add_input(array);
    task.add_input(array_copy);

    runtime->submit(std::move(task));
  }
}

}  // namespace test_io_hdf5_write
