/* Copyright 2023 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "library.h"
#include "core/mapping/mapping.h"
#include "world.h"

namespace rg {

class Mapper : public legate::mapping::LegateMapper {
 public:
  Mapper() {}

 private:
  Mapper(const Mapper& rhs)            = delete;
  Mapper& operator=(const Mapper& rhs) = delete;

  // Legate mapping functions
 public:
  void set_machine(const legate::mapping::MachineQueryInterface* machine) override
  {
    machine_ = machine;
  }

  legate::mapping::TaskTarget task_target(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::TaskTarget>& options) override
  {
    return options.front();
  }

  std::vector<legate::mapping::StoreMapping> store_mappings(
    const legate::mapping::Task& task,
    const std::vector<legate::mapping::StoreTarget>& options) override
  {
    return {};
  }

  legate::Scalar tunable_value(legate::TunableID tunable_id) override { return 0; }

 private:
  const legate::mapping::MachineQueryInterface* machine_;
};

static const char* const library_name = "registry";

Legion::Logger log_registry(library_name);

/*static*/ legate::TaskRegistrar& Registry::get_registrar()
{
  static legate::TaskRegistrar registrar;
  return registrar;
}

void registration_callback()
{
  auto context = legate::Runtime::get_runtime()->create_library(library_name);

  // Task registration via a registrar
  Registry::get_registrar().register_all_tasks(*context);
  // Immediate task registration
  WorldTask::register_variants(*context);

  context->register_mapper(std::make_unique<Mapper>(), 0);
}

}  // namespace rg

extern "C" {

void perform_registration(void) { legate::Core::perform_registration<rg::registration_callback>(); }
}
