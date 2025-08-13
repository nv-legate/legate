/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <legate/mapping/detail/mapping.h>
#include <legate/mapping/detail/store.h>
#include <legate/type/detail/types.h>

#include <gtest/gtest.h>

#include <utilities/utilities.h>

namespace store_mapping_unit {

namespace {

using StoreMappingTest = DefaultFixture;

// Helper class to manage Store lifetime
class TestStoreHolder {
 public:
  TestStoreHolder()
  {
    // Create a primitive int64 type
    auto type =
      legate::make_internal_shared<legate::detail::PrimitiveType>(legate::Type::Code::INT64);

    // Create a future store using FutureWrapper
    auto domain         = Legion::Domain{Legion::Rect<1>{0, 9}};
    auto future_wrapper = legate::mapping::detail::FutureWrapper{0, domain};

    detail_store_ = std::make_unique<legate::mapping::detail::Store>(1,  // dim
                                                                     std::move(type),
                                                                     std::move(future_wrapper));
  }

  legate::mapping::Store get_store() { return legate::mapping::Store{detail_store_.get()}; }

  // Get the internal shared pointer for the detail constructor
  legate::InternalSharedPtr<legate::mapping::detail::Store> get_detail_store_ptr()
  {
    return legate::InternalSharedPtr<legate::mapping::detail::Store>{detail_store_.get(),
                                                                     [](auto*) {}};
  }

 private:
  std::unique_ptr<legate::mapping::detail::Store> detail_store_;
};

// Helper class to create multiple stores
class MultipleStoreHolder {
 public:
  explicit MultipleStoreHolder(std::size_t count)
  {
    for (std::size_t i = 0; i < count; ++i) {
      holders_.emplace_back(std::make_unique<TestStoreHolder>());
    }
  }

  std::vector<legate::InternalSharedPtr<legate::mapping::detail::Store>> get_detail_store_ptrs()
  {
    std::vector<legate::InternalSharedPtr<legate::mapping::detail::Store>> ptrs;
    ptrs.reserve(holders_.size());
    for (auto& holder : holders_) {
      ptrs.push_back(holder->get_detail_store_ptr());
    }
    return ptrs;
  }

  std::vector<legate::mapping::Store> get_stores()
  {
    std::vector<legate::mapping::Store> stores;
    stores.reserve(holders_.size());
    for (auto& holder : holders_) {
      stores.push_back(holder->get_store());
    }
    return stores;
  }

 private:
  std::vector<std::unique_ptr<TestStoreHolder>> holders_;
};

}  // namespace

TEST_F(StoreMappingTest, CreateEmpty)
{
  auto empty_detail_mapping = std::make_unique<legate::mapping::detail::StoreMapping>();
  const legate::mapping::StoreMapping mapping{std::move(empty_detail_mapping)};

  ASSERT_EQ(mapping.stores().size(), 0);
  ASSERT_EQ(mapping.policy(), legate::mapping::InstanceMappingPolicy{});
}

TEST_F(StoreMappingTest, CreateDefaultMapping)
{
  auto store_holder = TestStoreHolder{};
  auto test_store   = store_holder.get_store();

  auto mapping = legate::mapping::StoreMapping::default_mapping(
    test_store, legate::mapping::StoreTarget::SYSMEM, false);

  ASSERT_EQ(mapping.policy().target, legate::mapping::StoreTarget::SYSMEM);
  ASSERT_EQ(mapping.policy().exact, false);
  ASSERT_EQ(mapping.stores().size(), 1);
}

TEST_F(StoreMappingTest, CreateMappingWithStore)
{
  auto store_holder = TestStoreHolder{};
  auto test_store   = store_holder.get_store();
  auto policy =
    legate::mapping::InstanceMappingPolicy{}.with_target(legate::mapping::StoreTarget::FBMEM);
  auto mapping = legate::mapping::StoreMapping::create(test_store, std::move(policy));

  ASSERT_EQ(mapping.policy().target, legate::mapping::StoreTarget::FBMEM);
  ASSERT_EQ(mapping.policy().exact, false);
  ASSERT_EQ(mapping.stores().size(), 1);
}

TEST_F(StoreMappingTest, CreateMappingWithStores)
{
  auto store_holder = TestStoreHolder{};
  auto test_store   = store_holder.get_store();
  auto policy =
    legate::mapping::InstanceMappingPolicy{}.with_target(legate::mapping::StoreTarget::FBMEM);
  auto mapping = legate::mapping::StoreMapping::create({test_store, test_store}, std::move(policy));

  ASSERT_EQ(mapping.policy().target, legate::mapping::StoreTarget::FBMEM);
  ASSERT_EQ(mapping.policy().exact, false);
  ASSERT_EQ(mapping.stores().size(), 2);
}

TEST_F(StoreMappingTest, CreateMappingWithStoresNegative)
{
  auto policy =
    legate::mapping::InstanceMappingPolicy{}.with_target(legate::mapping::StoreTarget::FBMEM);

  ASSERT_THAT(
    [&] {
      static_cast<void>(legate::mapping::StoreMapping::create(std::vector<legate::mapping::Store>{},
                                                              std::move(policy)));
    },
    ::testing::ThrowsMessage<std::invalid_argument>(
      ::testing::HasSubstr("Invalid to create a store mapping without any store")));
}

TEST_F(StoreMappingTest, CreateDetailMappingWithSingleStorePtr)
{
  auto store_holder = TestStoreHolder{};
  auto store_ptr    = store_holder.get_detail_store_ptr();

  auto policy = legate::mapping::InstanceMappingPolicy{}
                  .with_target(legate::mapping::StoreTarget::FBMEM)
                  .with_exact(true);

  std::vector<legate::InternalSharedPtr<legate::mapping::detail::Store>> store_ptrs = {store_ptr};
  auto detail_mapping = std::make_unique<legate::mapping::detail::StoreMapping>(
    std::move(policy),
    legate::Span<const legate::InternalSharedPtr<legate::mapping::detail::Store>>{store_ptrs});

  ASSERT_EQ(detail_mapping->stores().size(), 1);
  ASSERT_EQ(detail_mapping->policy().target, legate::mapping::StoreTarget::FBMEM);
  ASSERT_EQ(detail_mapping->policy().exact, true);

  // Create public mapping from detail mapping
  legate::mapping::StoreMapping mapping{std::move(detail_mapping)};
  ASSERT_EQ(mapping.stores().size(), 1);
  ASSERT_EQ(mapping.policy().target, legate::mapping::StoreTarget::FBMEM);
  ASSERT_EQ(mapping.policy().exact, true);
}

TEST_F(StoreMappingTest, CreateDetailMappingWithMultipleStorePtrs)
{
  constexpr auto num_stores = 3;
  auto multi_holder         = MultipleStoreHolder{num_stores};
  auto store_ptrs           = multi_holder.get_detail_store_ptrs();

  auto policy = legate::mapping::InstanceMappingPolicy{}
                  .with_target(legate::mapping::StoreTarget::ZCMEM)
                  .with_allocation_policy(legate::mapping::AllocPolicy::MUST_ALLOC);

  auto detail_mapping = std::make_unique<legate::mapping::detail::StoreMapping>(
    std::move(policy),
    legate::Span<const legate::InternalSharedPtr<legate::mapping::detail::Store>>{store_ptrs});

  ASSERT_EQ(detail_mapping->stores().size(), num_stores);
  ASSERT_EQ(detail_mapping->policy().target, legate::mapping::StoreTarget::ZCMEM);
  ASSERT_EQ(detail_mapping->policy().allocation, legate::mapping::AllocPolicy::MUST_ALLOC);

  // Verify that InternalSharedPtr<Store> is correctly converted to Store* pointers
  auto mapped_stores = detail_mapping->stores();
  for (std::size_t i = 0; i < num_stores; ++i) {
    ASSERT_EQ(mapped_stores[i], store_ptrs[i].get());
  }

  // Create public mapping from detail mapping
  legate::mapping::StoreMapping mapping{std::move(detail_mapping)};
  ASSERT_EQ(mapping.stores().size(), num_stores);
  ASSERT_EQ(mapping.policy().target, legate::mapping::StoreTarget::ZCMEM);
  ASSERT_EQ(mapping.policy().allocation, legate::mapping::AllocPolicy::MUST_ALLOC);
}

TEST_F(StoreMappingTest, CreateDetailMappingWithEmptyStorePtrs)
{
  auto policy =
    legate::mapping::InstanceMappingPolicy{}.with_target(legate::mapping::StoreTarget::SOCKETMEM);

  std::vector<legate::InternalSharedPtr<legate::mapping::detail::Store>> empty_store_ptrs;
  auto detail_mapping = std::make_unique<legate::mapping::detail::StoreMapping>(
    std::move(policy),
    legate::Span<const legate::InternalSharedPtr<legate::mapping::detail::Store>>{
      empty_store_ptrs});

  ASSERT_EQ(detail_mapping->stores().size(), 0);
  ASSERT_EQ(detail_mapping->policy().target, legate::mapping::StoreTarget::SOCKETMEM);

  // Create public mapping from detail mapping
  legate::mapping::StoreMapping mapping{std::move(detail_mapping)};
  ASSERT_EQ(mapping.stores().size(), 0);
  ASSERT_EQ(mapping.policy().target, legate::mapping::StoreTarget::SOCKETMEM);
}

TEST_F(StoreMappingTest, RequirementIndexEmptyStores)
{
  auto empty_detail_mapping       = std::make_unique<legate::mapping::detail::StoreMapping>();
  constexpr auto EXPECTED_INVALID = static_cast<std::uint32_t>(-1);

  ASSERT_EQ(empty_detail_mapping->requirement_index(), EXPECTED_INVALID);
}

TEST_F(StoreMappingTest, RequirementIndicesWithFutureStores)
{
  auto future_store_holder = TestStoreHolder{};
  auto future_store_ptr    = future_store_holder.get_detail_store_ptr();

  ASSERT_TRUE(future_store_ptr->is_future());

  auto policy = legate::mapping::InstanceMappingPolicy{};
  std::vector<legate::InternalSharedPtr<legate::mapping::detail::Store>> store_ptrs = {
    future_store_ptr};
  auto detail_mapping = std::make_unique<legate::mapping::detail::StoreMapping>(
    std::move(policy),
    legate::Span<const legate::InternalSharedPtr<legate::mapping::detail::Store>>{store_ptrs});
  auto indices = detail_mapping->requirement_indices();

  ASSERT_TRUE(indices.empty());
}

TEST_F(StoreMappingTest, RequirementsWithFutureStores)
{
  auto future_store_holder = TestStoreHolder{};
  auto future_store_ptr    = future_store_holder.get_detail_store_ptr();

  ASSERT_TRUE(future_store_ptr->is_future());

  auto policy = legate::mapping::InstanceMappingPolicy{};
  std::vector<legate::InternalSharedPtr<legate::mapping::detail::Store>> store_ptrs = {
    future_store_ptr};
  auto detail_mapping = std::make_unique<legate::mapping::detail::StoreMapping>(
    std::move(policy),
    legate::Span<const legate::InternalSharedPtr<legate::mapping::detail::Store>>{store_ptrs});
  auto requirements = detail_mapping->requirements();

  ASSERT_TRUE(requirements.empty());
}

TEST_F(StoreMappingTest, RequirementsWithMultipleFutureStores)
{
  constexpr auto num_stores = 2;
  auto multi_holder         = MultipleStoreHolder{num_stores};
  auto store_ptrs           = multi_holder.get_detail_store_ptrs();

  for (const auto& store_ptr : store_ptrs) {
    ASSERT_TRUE(store_ptr->is_future());
  }

  auto policy         = legate::mapping::InstanceMappingPolicy{};
  auto detail_mapping = std::make_unique<legate::mapping::detail::StoreMapping>(
    std::move(policy),
    legate::Span<const legate::InternalSharedPtr<legate::mapping::detail::Store>>{store_ptrs});
  auto requirements = detail_mapping->requirements();

  ASSERT_TRUE(requirements.empty());
}

}  // namespace store_mapping_unit
