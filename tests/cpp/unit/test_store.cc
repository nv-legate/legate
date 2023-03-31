#include <iomanip>
#include <iostream>
#include <sstream>
#include <valarray>

#include "legate.h"

namespace unit {

void test_store_creation()
{
  // Bound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store({4, 4}, legate::LegateTypeCode::INT64_LT);
    assert(!store.unbound());
    assert(store.dim() == 2);
    assert((store.extents() == std::vector<size_t>{4, 4}));
    assert(store.code() == legate::LegateTypeCode::INT64_LT);
    assert(!store.transformed());
  }

  // Unbound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store(legate::LegateTypeCode::INT64_LT);
    assert(store.unbound());
    assert(store.dim() == 1);
    assert(store.code() == legate::LegateTypeCode::INT64_LT);
    assert(!store.transformed());
    // TODO: How can we do negative test cases?
    // with pytest.raises(ValueError):
    //     store.shape
  }
}

void test_store_valid_transform()
{
  // Bound
  auto runtime = legate::Runtime::get_runtime();
  auto store   = runtime->create_store({4, 3}, legate::LegateTypeCode::INT64_LT);

  auto promoted = store.promote(0, 5);
  assert((promoted.extents() == std::vector<size_t>{5, 4, 3}));
  assert(promoted.transformed());

  auto projected = store.project(0, 1);
  assert((projected.extents() == std::vector<size_t>{
                                   3,
                                 }));
  assert(projected.transformed());

  auto sliced = store.slice(1, std::slice(1, 3, 1));
  // TODO: Enable once implemented
  // assert((sliced.extents() == std::vector<size_t>{4, 2}));
  // assert(sliced.transformed());

  auto transposed = store.transpose({1, 0});
  // TODO: Enable once implemented
  // assert((transposed.extents() == std::vector<size_t>{3, 4}));
  // assert(transposed.transformed());

  auto delinearized = store.delinearize(0, {2, 2});
  // TODO: Enable once implemented
  // assert((delinearized.extents() == std::vector<size_t>{2, 2, 3}));
  // assert(delinearized.transformed());
}

void test_store_invalid_transform()
{
  // Bound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store({4, 3}, legate::LegateTypeCode::INT64_LT);

    // with pytest.raises(ValueError):
    //     store.promote(3, 5)

    // with pytest.raises(ValueError):
    //     store.promote(-3, 5)

    // with pytest.raises(ValueError):
    //     store.project(2, 1)

    // with pytest.raises(ValueError):
    //     store.project(-3, 1)

    // with pytest.raises(ValueError):
    //     store.project(0, 4)

    // with pytest.raises(ValueError):
    //     store.slice(2, slice(1, 3))

    // with pytest.raises(NotImplementedError):
    //     store.slice(1, slice(1, 3, 2))

    // with pytest.raises(ValueError):
    //     store.slice(1, slice(1, 4))

    // with pytest.raises(ValueError):
    //     store.transpose((2,))

    // with pytest.raises(ValueError):
    //     store.transpose((0, 0))

    // with pytest.raises(ValueError):
    //     store.transpose((2, 0))

    // with pytest.raises(ValueError):
    //     store.delinearize(2, (2, 3))

    // with pytest.raises(ValueError):
    //     store.delinearize(0, (2, 3))
  }

  // Unbound
  {
    auto runtime = legate::Runtime::get_runtime();
    auto store   = runtime->create_store(legate::LegateTypeCode::INT64_LT);
    // with pytest.raises(ValueError):
    //     store.promote(1, 1)
  }
}

void legate_main(int32_t argc, char** argv)
{
  auto runtime = legate::Runtime::get_runtime();

  test_store_creation();
  test_store_valid_transform();
  test_store_invalid_transform();
}

}  // namespace unit

int main(int argc, char** argv)
{
  legate::initialize(argc, argv);

  legate::set_main_function(unit::legate_main);

  return legate::start(argc, argv);
}
