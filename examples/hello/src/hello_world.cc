#include "hello_world.h"
#include "legate_library.h"

namespace hello {

Legion::Logger logger("legate.hello");

class HelloWorldTask : public Task<HelloWorldTask, HELLO_WORLD> {
public:
  static void cpu_variant(legate::TaskContext &context) {
    std::string message = context.scalars()[0].value<std::string>();
    std::cout << message << std::endl;
  }
};

} // namespace hello

namespace // unnamed
{

static void __attribute__((constructor)) register_tasks(void) {
  hello::HelloWorldTask::register_variants();
}

} // namespace
