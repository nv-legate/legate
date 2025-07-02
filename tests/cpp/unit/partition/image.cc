/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/data/detail/logical_store.h>
#include <legate/partitioning/detail/partition.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace unit {

class ImageTest : public DefaultFixture {
 public:
  void SetUp() override
  {
    DefaultFixture::SetUp();
    image = create_image();
  }

  [[nodiscard]] legate::InternalSharedPtr<legate::detail::Image> create_image()
  {
    auto runtime   = legate::Runtime::get_runtime();
    auto store     = runtime->create_store(legate::Shape{1}, legate::int32());
    auto partition = legate::detail::create_tiling(legate::tuple<std::uint64_t>{1},
                                                   legate::tuple<std::uint64_t>{1});
    return legate::detail::create_image(store.impl(),
                                        partition,
                                        legate::mapping::detail::Machine{},
                                        legate::ImageComputationHint::NO_HINT);
  }

  legate::InternalSharedPtr<legate::detail::Image> image;
};

TEST_F(ImageTest, Kind) { ASSERT_EQ(image->kind(), legate::detail::Partition::Kind::IMAGE); }

TEST_F(ImageTest, Compare)
{
  auto image1 = legate::detail::Image{*image};
  ASSERT_EQ(image1, *image);

  auto image2 = create_image();
  ASSERT_FALSE(*image2 == *image);
}

TEST_F(ImageTest, IsDisjointFor)
{
  auto domain1 = Legion::Domain::NO_DOMAIN;
  ASSERT_TRUE(image->is_disjoint_for(domain1));

  auto domain2 = Legion::Domain{Legion::Rect<1>{0, 1}};
  ASSERT_FALSE(image->is_disjoint_for(domain2));
}

TEST_F(ImageTest, ColorShape)
{
  auto expected_color_shape = legate::tuple<std::uint64_t>{1};
  ASSERT_EQ(image->color_shape(), expected_color_shape);
}

TEST_F(ImageTest, IsCompleteFor)
{
  // This store itself does not actually matter, we just need to create one in order to get a
  // `Storage` reference.
  const auto store =
    legate::Runtime::get_runtime()->create_store(legate::Shape{1}, legate::int32());

  ASSERT_FALSE(image->is_complete_for(*store.impl()->get_storage()));
}

TEST_F(ImageTest, IsConvertible) { ASSERT_FALSE(image->is_convertible()); }

TEST_F(ImageTest, LaunchDomain)
{
  ASSERT_TRUE(image->has_launch_domain());
  auto domain = Legion::Domain{Legion::Rect<1>{0, 0}};
  ASSERT_EQ(image->launch_domain(), domain);
}

TEST_F(ImageTest, SatisfiesRestrictions)
{
  auto restrictions1 = legate::detail::Restrictions{legate::detail::Restriction::ALLOW};
  ASSERT_TRUE(image->satisfies_restrictions(restrictions1));

  auto restrictions2 = legate::detail::Restrictions{legate::detail::Restriction::AVOID};
  ASSERT_TRUE(image->satisfies_restrictions(restrictions2));

  auto restrictions3 = legate::detail::Restrictions{legate::detail::Restriction::FORBID};
  ASSERT_TRUE(image->satisfies_restrictions(restrictions3));
}

TEST_F(ImageTest, SatisfiesRestrictionsNegative)
{
  if (!LEGATE_DEFINED(LEGATE_USE_DEBUG)) {
    GTEST_SKIP() << "Sizes are only checked in debug builds";
  }

  auto restrictions1 = legate::detail::Restrictions{legate::detail::Restriction::ALLOW,
                                                    legate::detail::Restriction::AVOID};
  ASSERT_THROW(static_cast<void>(image->satisfies_restrictions(restrictions1)),
               std::invalid_argument);

  auto restrictions2 = legate::detail::Restrictions{};
  ASSERT_THROW(static_cast<void>(image->satisfies_restrictions(restrictions2)),
               std::invalid_argument);
}

TEST_F(ImageTest, Scale)
{
  // Scale is not implemented for image partitions
  auto factors = legate::tuple<std::uint64_t>{2, 2};
  ASSERT_THROW(static_cast<void>(image->scale(factors)), std::runtime_error);
}

TEST_F(ImageTest, Bloat)
{
  // Bloat is not implemented for image partitions
  auto low_offsets  = legate::tuple<std::uint64_t>{0, 0};
  auto high_offsets = legate::tuple<std::uint64_t>{1, 1};
  ASSERT_THROW(static_cast<void>(image->bloat(low_offsets, high_offsets)), std::runtime_error);
}

TEST_F(ImageTest, Convert)
{
  // Test identity transformation
  auto transform1 = legate::make_internal_shared<legate::detail::TransformStack>();
  auto image1     = image->convert(image, transform1);
  auto image1_raw = image1.get();
  auto image_raw  = image.get();
  ASSERT_EQ(image1_raw, image_raw);

  // Test non-identity transformation
  auto extra_dim  = 1;
  auto dim_size   = 1;
  auto transform2 = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Promote>(extra_dim, dim_size), std::move(transform1));
  ASSERT_THROW(static_cast<void>(image->convert(image, transform2)),
               legate::detail::NonInvertibleTransformation);
}

TEST_F(ImageTest, Invert)
{
  // Test identity transformation
  auto transform1 = legate::make_internal_shared<legate::detail::TransformStack>();
  auto image1     = image->invert(image, transform1);
  auto image1_raw = image1.get();
  auto image_raw  = image.get();
  ASSERT_EQ(image1_raw, image_raw);

  // Test non-identity transformation
  auto dim        = 1;
  auto sizes      = std::vector<std::uint64_t>{1};
  auto transform2 = legate::make_internal_shared<legate::detail::TransformStack>(
    std::make_unique<legate::detail::Delinearize>(dim, std::move(sizes)), std::move(transform1));
  ASSERT_THROW(static_cast<void>(image->invert(image, transform2)),
               legate::detail::NonInvertibleTransformation);
}

TEST_F(ImageTest, ToString)
{
  ASSERT_THAT(image->to_string(),
              ::testing::MatchesRegex(
                R"(Image\(func: Store\([0-9]+\) .*, partition: Tiling\(.*\), hint: NO_HINT\))"));
}

}  // namespace unit
