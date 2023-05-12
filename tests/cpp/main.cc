#include <gtest/gtest.h>
#include "legate.h"

class Environment : public ::testing::Environment {
 public:
  Environment(int argc, char** argv) : argc_(argc), argv_(argv) {}

  void SetUp() override
  {
    legate::initialize(argc_, argv_);
    EXPECT_EQ(legate::start(argc_, argv_), 0);
  }
  void TearDown() override { EXPECT_EQ(legate::wait_for_shutdown(), 0); }

 private:
  int argc_;
  char** argv_;
};

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  ::testing::AddGlobalTestEnvironment(new Environment(argc, argv));

  return RUN_ALL_TESTS();
}
