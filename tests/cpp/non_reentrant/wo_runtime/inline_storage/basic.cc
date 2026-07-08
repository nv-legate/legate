/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/data/detail/storage.h>
#include <legate/redop/redop.h>
#include <legate/runtime/detail/runtime.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <utilities/env.h>
#include <utilities/utilities.h>

namespace test_inline_storage_unit {

namespace {

class InlineStorageUnit : public DefaultFixture {
 protected:
  void SetUp() override
  {
    ASSERT_NO_THROW(legate::start());
    DefaultFixture::SetUp();
  }

  void TearDown() override
  {
    DefaultFixture::TearDown();
    ASSERT_EQ(legate::finish(), 0);
  }

 private:
  legate::test::Environment::TemporaryEnvVar legate_config_{/* name */ "LEGATE_CONFIG",
                                                            /* value */ "--inline-task-launch 1",
                                                            /* overwrite */ true};
};

void assert_int32_store_values(const legate::LogicalStore& store,
                               std::int32_t expected,
                               std::size_t expected_size)
{
  constexpr auto DIM = std::int32_t{1};
  const auto phys    = store.get_physical_store();
  const auto mdspan  = phys.span_read_accessor<std::int32_t, DIM>();

  ASSERT_EQ(mdspan.extent(0), expected_size);
  for (auto idx = 0; idx < mdspan.extent(0); ++idx) {
    ASSERT_EQ(mdspan(idx), expected);
  }
}

}  // namespace

TEST_F(InlineStorageUnit, Basic)
{
  constexpr auto DIM   = 2;
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{DIM, DIM}, legate::int32());
  constexpr auto VALUE = std::int32_t{32};

  runtime->issue_fill(store, legate::Scalar{VALUE});

  auto&& impl = store.impl();

  ASSERT_EQ(impl->get_storage()->kind(), legate::detail::Storage::Kind::INLINE_STORAGE);

  const auto phys = store.get_physical_store();

  ASSERT_TRUE(phys.valid());
  ASSERT_FALSE(phys.is_partitioned());

  const auto mdspan = phys.span_read_accessor<std::int32_t, DIM>();

  for (auto i = 0; i < mdspan.extent(0); ++i) {
    for (auto j = 0; j < mdspan.extent(1); ++j) {
      ASSERT_EQ(mdspan(i, j), VALUE);
    }
  }
}

TEST_F(InlineStorageUnit, Copy)
{
  constexpr auto NUM_ELEMENTS = std::size_t{6};
  constexpr auto FILL_VALUE   = std::int32_t{7};
  auto* const runtime         = legate::Runtime::get_runtime();
  auto source                 = runtime->create_store(legate::Shape{NUM_ELEMENTS}, legate::int32());
  auto target                 = runtime->create_store(legate::Shape{NUM_ELEMENTS}, legate::int32());

  runtime->issue_fill(source, legate::Scalar{FILL_VALUE});
  runtime->issue_copy(target, source);

  assert_int32_store_values(target, FILL_VALUE, NUM_ELEMENTS);
}

TEST_F(InlineStorageUnit, CopyFromInlineSourceToRegionTarget)
{
  constexpr auto NUM_ELEMENTS = std::size_t{6};
  constexpr auto FILL_VALUE   = std::int32_t{11};
  auto* const runtime         = legate::Runtime::get_runtime();
  const auto shape            = legate::Shape{NUM_ELEMENTS};
  const auto public_type      = legate::int32();
  const auto type             = legate::InternalSharedPtr<legate::detail::Type>{public_type.impl()};
  auto source                 = runtime->create_store(shape, public_type);
  auto region_field =
    runtime->impl()->create_region_field(shape.impl(), static_cast<std::uint32_t>(type->size()));
  auto storage = legate::make_internal_shared<legate::detail::Storage>(
    shape.impl(), std::move(region_field), "");
  auto target = legate::LogicalStore{
    legate::make_internal_shared<legate::detail::LogicalStore>(std::move(storage), type)};

  ASSERT_EQ(source.impl()->get_storage()->kind(), legate::detail::Storage::Kind::INLINE_STORAGE);
  ASSERT_EQ(target.impl()->get_storage()->kind(), legate::detail::Storage::Kind::REGION_FIELD);

  runtime->issue_fill(source, legate::Scalar{FILL_VALUE});
  runtime->issue_copy(target, source);

  assert_int32_store_values(target, FILL_VALUE, NUM_ELEMENTS);
}

TEST_F(InlineStorageUnit, Scalar)
{
  constexpr std::int32_t DIM = 1;
  const std::int32_t VALUE   = 2;
  const std::size_t DIM_SIZE = 10;
  const legate::Scalar scalar{VALUE};
  auto* const runtime = legate::Runtime::get_runtime();

  auto store = runtime->create_store(scalar);

  auto broadcasted = store.broadcast(/*dim=*/0, DIM_SIZE);

  const auto phys   = broadcasted.get_physical_store();
  const auto mdspan = phys.span_read_accessor<std::int32_t, DIM>();

  ASSERT_EQ(mdspan.extent(0), 10);

  for (auto i = 0; i < mdspan.extent(0); i++) {
    ASSERT_EQ(mdspan(i), VALUE);
  }
}

TEST_F(InlineStorageUnit, Slice)
{
  constexpr auto DIM   = 2;
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{2, 3}, legate::int32());
  constexpr auto VALUE = std::int32_t{32};

  runtime->issue_fill(store, legate::Scalar{VALUE});

  const auto first_row     = store.slice(/*dim=*/0, legate::Slice{legate::Slice::OPEN, /*stop=*/1});
  constexpr auto NEW_VALUE = 2 * VALUE;

  {
    const auto mdspan = first_row.get_physical_store().span_write_accessor<std::int32_t, DIM>();

    ASSERT_EQ(mdspan.extent(0), 1);
    ASSERT_EQ(mdspan.extent(1), 3);

    for (auto i = 0; i < mdspan.extent(0); ++i) {
      for (auto j = 0; j < mdspan.extent(1); ++j) {
        mdspan(i, j) = NEW_VALUE;
      }
    }
  }

  const auto phys   = store.get_physical_store();
  const auto mdspan = phys.span_read_accessor<std::int32_t, DIM>();

  for (auto i = 0; i < mdspan.extent(0); ++i) {
    for (auto j = 0; j < mdspan.extent(1); ++j) {
      if (i < 1) {
        ASSERT_EQ(mdspan(i, j), NEW_VALUE)
          << "mspan(" << i << ", " << j << ") = " << mdspan(i, j) << " != " << NEW_VALUE;
      } else {
        ASSERT_EQ(mdspan(i, j), VALUE)
          << "mspan(" << i << ", " << j << ") = " << mdspan(i, j) << " != " << VALUE;
      }
    }
  }
}

TEST_F(InlineStorageUnit, Project)
{
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{2, 3}, legate::int32());
  constexpr auto VALUE = std::int32_t{32};

  runtime->issue_fill(store, legate::Scalar{VALUE});

  const auto last_column   = store.project(/*dim=*/1, /*index=*/2);
  constexpr auto NEW_VALUE = 2 * VALUE;

  {
    const auto mdspan = last_column.get_physical_store().span_write_accessor<std::int32_t, 1>();

    ASSERT_EQ(mdspan.extent(0), 2);

    for (auto i = 0; i < mdspan.extent(0); ++i) {
      mdspan(i) = NEW_VALUE;
    }
  }

  const auto phys   = store.get_physical_store();
  const auto mdspan = phys.span_read_accessor<std::int32_t, 2>();

  for (auto i = 0; i < mdspan.extent(0); ++i) {
    for (auto j = 0; j < mdspan.extent(1); ++j) {
      if (j < 2) {
        ASSERT_EQ(mdspan(i, j), VALUE)
          << "mspan(" << i << ", " << j << ") = " << mdspan(i, j) << " != " << VALUE;
      } else {
        ASSERT_EQ(mdspan(i, j), NEW_VALUE)
          << "mspan(" << i << ", " << j << ") = " << mdspan(i, j) << " != " << NEW_VALUE;
      }
    }
  }
}

TEST_F(InlineStorageUnit, Promote)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto store    = runtime->create_store(legate::Shape{3}, legate::int32());
  constexpr auto VALUE{std::int32_t{32}};

  runtime->issue_fill(store, legate::Scalar{VALUE});

  const auto promoted = store.promote(/*extra_dim=*/0, /*dim_size=*/2);
  constexpr auto NEW_VALUE{2 * VALUE};

  {
    const auto mdspan = promoted.get_physical_store().span_write_accessor<std::int32_t, 2>();

    ASSERT_EQ(mdspan.extent(0), 2);
    ASSERT_EQ(mdspan.extent(1), 3);

    for (auto j = 0; j < mdspan.extent(1); ++j) {
      mdspan(1, j) = NEW_VALUE;
    }
  }

  const auto phys   = store.get_physical_store();
  const auto mdspan = phys.span_read_accessor<std::int32_t, 1>();

  for (auto i = 0; i < mdspan.extent(0); ++i) {
    ASSERT_EQ(mdspan(i), NEW_VALUE) << "mspan(" << i << ") = " << mdspan(i) << " != " << NEW_VALUE;
  }
}

TEST_F(InlineStorageUnit, Broadcast)
{
  auto* const runtime = legate::Runtime::get_runtime();
  const auto store    = runtime->create_store(legate::Shape{1, 3}, legate::int32());
  constexpr auto VALUE{std::int32_t{32}};

  runtime->issue_fill(store, legate::Scalar{VALUE});

  const auto broadcasted = store.broadcast(/*dim=*/0, /*dim_size=*/2);
  constexpr auto NEW_VALUE{2 * VALUE};

  {
    const auto mdspan = broadcasted.get_physical_store().span_write_accessor<std::int32_t, 2>();

    ASSERT_EQ(mdspan.extent(0), 2);
    ASSERT_EQ(mdspan.extent(1), 3);

    for (auto j = 0; j < mdspan.extent(1); ++j) {
      mdspan(1, j) = NEW_VALUE;
    }
  }

  const auto phys   = store.get_physical_store();
  const auto mdspan = phys.span_read_accessor<std::int32_t, 2>();

  for (auto j = 0; j < mdspan.extent(1); ++j) {
    ASSERT_EQ(mdspan(0, j), NEW_VALUE)
      << "mspan(0, " << j << ") = " << mdspan(0, j) << " != " << NEW_VALUE;
  }

  const auto bcast_span = broadcasted.get_physical_store().span_read_accessor<std::int32_t, 2>();

  for (auto i = 0; i < bcast_span.extent(0); ++i) {
    for (auto j = 0; j < bcast_span.extent(1); ++j) {
      ASSERT_EQ(bcast_span(i, j), NEW_VALUE)
        << "mspan(" << i << ", " << j << ") = " << bcast_span(i, j) << " != " << NEW_VALUE;
    }
  }
}

TEST_F(InlineStorageUnit, FillAndCopy)
{
  auto* const runtime = legate::Runtime::get_runtime();
  auto store1         = runtime->create_store(legate::Shape{5}, legate::int32());
  auto store2         = runtime->create_store(legate::Shape{5}, legate::int32());

  constexpr std::int32_t VALUE1 = 123;
  constexpr std::int32_t VALUE2 = 555;

  runtime->issue_fill(store1, legate::Scalar{VALUE1});
  runtime->issue_fill(store2, legate::Scalar{VALUE2});

  {
    const auto span1 = store1.get_physical_store().span_read_accessor<std::int32_t, 1>();
    const auto span2 = store2.get_physical_store().span_read_accessor<std::int32_t, 1>();
    for (auto i = 0; i < span1.extent(0); ++i) {
      ASSERT_EQ(span1(i), VALUE1) << "span1(" << i << ") = " << span1(i) << " != " << VALUE1;
      ASSERT_EQ(span2(i), VALUE2) << "span2(" << i << ") = " << span2(i) << " != " << VALUE2;
    }
  }

  runtime->issue_copy(store1, store2);

  {
    const auto span2 = store2.get_physical_store().span_read_accessor<std::int32_t, 1>();
    for (auto i = 0; i < span2.extent(0); ++i) {
      ASSERT_EQ(span2(i), VALUE2) << "After copy, span2(" << i << ") = " << span2(i)
                                  << " != " << VALUE2;
    }
  }

  {
    const auto span1 = store1.get_physical_store().span_read_accessor<std::int32_t, 1>();
    for (auto i = 0; i < span1.extent(0); ++i) {
      ASSERT_EQ(span1(i), VALUE2) << "After copy, span1(" << i << ") = " << span1(i)
                                  << " != " << VALUE2;
    }
  }
}

TEST_F(InlineStorageUnit, ExternalAllocation)
{
  constexpr std::size_t SIZE = 7;
  const std::int64_t VAL     = 100;
  auto* mem                  = new std::int64_t[SIZE];

  for (std::size_t i = 0; i < SIZE; ++i) {
    mem[i] = VAL + static_cast<std::int64_t>(i);
  }

  const legate::ExternalAllocation ext_alloc = legate::ExternalAllocation::create_sysmem(
    mem,
    SIZE * sizeof(std::int64_t),
    /*read_only=*/false,
    [](void* ptr) { delete[] static_cast<std::int64_t*>(ptr); });

  auto* const runtime = legate::Runtime::get_runtime();
  const legate::LogicalStore store =
    runtime->create_store(legate::Shape{SIZE},  // one-dimensional store
                          legate::int64(),
                          ext_alloc);

  auto span = store.get_physical_store().span_read_accessor<std::int64_t, 1>();
  for (std::size_t i = 0; i < SIZE; ++i) {
    ASSERT_EQ(span(i), VAL + i) << "span(" << i << ") = " << span(i) << " != " << (VAL + i);
  }
}

TEST_F(InlineStorageUnit, RemapSysmemToSocketmem)
{
  constexpr auto DIM   = 2;
  constexpr auto VALUE = std::int32_t{9};
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{DIM, DIM}, legate::int32());

  runtime->issue_fill(store, legate::Scalar{VALUE});

  const auto phys_sys = store.get_physical_store(legate::mapping::StoreTarget::SYSMEM);

  ASSERT_NE(phys_sys.get_inline_allocation().ptr, nullptr);

  const auto phys_sock = store.get_physical_store(legate::mapping::StoreTarget::SOCKETMEM);

  ASSERT_NE(phys_sock.get_inline_allocation().ptr, nullptr);

  const auto mdspan = phys_sock.span_read_accessor<std::int32_t, DIM>();

  for (auto i = 0; i < mdspan.extent(0); ++i) {
    for (auto j = 0; j < mdspan.extent(1); ++j) {
      ASSERT_EQ(mdspan(i, j), VALUE);
    }
  }
}

TEST_F(InlineStorageUnit, GetInlineAllocationTransformed)
{
  constexpr auto DIM   = 2;
  constexpr auto VALUE = std::int32_t{55};
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{DIM, DIM}, legate::int32());

  runtime->issue_fill(store, legate::Scalar{VALUE});

  auto promoted   = store.promote(/*extra_dim=*/0, /*dim_size=*/1);
  const auto phys = promoted.get_physical_store();

  ASSERT_TRUE(phys.transformed());

  const auto alloc = phys.get_inline_allocation();

  ASSERT_NE(alloc.ptr, nullptr);
  ASSERT_EQ(alloc.strides.size(), static_cast<std::size_t>(promoted.dim()));
}

TEST_F(InlineStorageUnit, ToLogicalStoreThrows)
{
  constexpr auto DIM   = 2;
  constexpr auto VALUE = std::int32_t{11};
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{DIM, DIM}, legate::int32());

  runtime->issue_fill(store, legate::Scalar{VALUE});

  const auto phys = store.get_physical_store();

  ASSERT_THAT([&] { static_cast<void>(phys.to_logical_store()); },
              ::testing::ThrowsMessage<std::runtime_error>(
                ::testing::HasSubstr("does not support to_logical_store()")));
}

TEST_F(InlineStorageUnit, TransformedReadAccessor)
{
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{3}, legate::int32());
  constexpr auto VALUE = std::int32_t{42};

  runtime->issue_fill(store, legate::Scalar{VALUE});

  const auto promoted = store.promote(/*extra_dim=*/0, /*dim_size=*/2);
  const auto phys     = promoted.get_physical_store();

  ASSERT_TRUE(phys.transformed());

  const auto acc   = phys.read_accessor<std::int32_t, 2>();
  const auto shape = phys.shape<2>();

  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    ASSERT_EQ(acc[*it], VALUE);
  }
}

TEST_F(InlineStorageUnit, TransformedWriteAccessor)
{
  auto* const runtime  = legate::Runtime::get_runtime();
  const auto store     = runtime->create_store(legate::Shape{3}, legate::int32());
  constexpr auto VALUE = std::int32_t{7};

  runtime->issue_fill(store, legate::Scalar{std::int32_t{0}});

  const auto promoted = store.promote(/*extra_dim=*/0, /*dim_size=*/2);
  const auto phys     = promoted.get_physical_store();

  ASSERT_TRUE(phys.transformed());

  {
    const auto acc   = phys.write_accessor<std::int32_t, 2>();
    const auto shape = phys.shape<2>();

    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      acc[*it] = VALUE;
    }
  }

  {
    const auto read_acc = phys.read_accessor<std::int32_t, 2>();
    const auto shape    = phys.shape<2>();

    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      ASSERT_EQ(read_acc[*it], VALUE);
    }
  }
}

TEST_F(InlineStorageUnit, TransformedReadWriteAccessor)
{
  auto* const runtime       = legate::Runtime::get_runtime();
  const auto store          = runtime->create_store(legate::Shape{3}, legate::int32());
  constexpr auto INIT_VALUE = std::int32_t{10};
  constexpr auto DELTA      = std::int32_t{5};

  runtime->issue_fill(store, legate::Scalar{INIT_VALUE});

  const auto promoted = store.promote(/*extra_dim=*/0, /*dim_size=*/2);
  const auto phys     = promoted.get_physical_store();

  ASSERT_TRUE(phys.transformed());

  const auto acc   = phys.read_write_accessor<std::int32_t, 2>();
  const auto shape = phys.shape<2>();

  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    ASSERT_EQ(acc[*it], INIT_VALUE);
  }

  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    acc[*it] = acc[*it] + DELTA;
  }

  // promote(extra_dim=0, dim_size=2) aliases (0,j) and (1,j) to the same underlying element,
  // so each element receives DELTA twice when iterating the full 2D shape.
  for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
    ASSERT_EQ(acc[*it], INIT_VALUE + (2 * DELTA));
  }
}

TEST_F(InlineStorageUnit, ReduceAccessor)
{
  auto* const runtime       = legate::Runtime::get_runtime();
  const auto store          = runtime->create_store(legate::Shape{2, 3}, legate::int32());
  constexpr auto INIT_VALUE = std::int32_t{10};
  constexpr auto RED_VALUE  = std::int32_t{5};

  runtime->issue_fill(store, legate::Scalar{INIT_VALUE});

  const auto phys = store.get_physical_store();

  ASSERT_FALSE(phys.transformed());

  {
    const auto acc   = phys.reduce_accessor<legate::SumReduction<std::int32_t>, true, 2>();
    const auto shape = phys.shape<2>();

    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      acc.reduce(*it, RED_VALUE);
    }
  }

  {
    const auto read_acc = phys.read_accessor<std::int32_t, 2>();
    const auto shape    = phys.shape<2>();

    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      ASSERT_EQ(read_acc[*it], INIT_VALUE + RED_VALUE);
    }
  }
}

TEST_F(InlineStorageUnit, TransformedReduceAccessor)
{
  auto* const runtime       = legate::Runtime::get_runtime();
  const auto store          = runtime->create_store(legate::Shape{3}, legate::int32());
  constexpr auto INIT_VALUE = std::int32_t{10};
  constexpr auto RED_VALUE  = std::int32_t{5};

  runtime->issue_fill(store, legate::Scalar{INIT_VALUE});

  const auto promoted = store.promote(/*extra_dim=*/0, /*dim_size=*/2);
  const auto phys     = promoted.get_physical_store();

  ASSERT_TRUE(phys.transformed());

  {
    const auto acc   = phys.reduce_accessor<legate::SumReduction<std::int32_t>, true, 2>();
    const auto shape = phys.shape<2>();

    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      acc.reduce(*it, RED_VALUE);
    }
  }

  {
    const auto read_acc = phys.read_accessor<std::int32_t, 2>();
    const auto shape    = phys.shape<2>();

    // promote(extra_dim=0, dim_size=2) aliases (0,j) and (1,j) to the same underlying element,
    // so each element receives RED_VALUE twice when iterating the full 2D shape.
    for (legate::PointInRectIterator<2> it{shape}; it.valid(); ++it) {
      ASSERT_EQ(read_acc[*it], INIT_VALUE + (2 * RED_VALUE));
    }
  }
}

}  // namespace test_inline_storage_unit
