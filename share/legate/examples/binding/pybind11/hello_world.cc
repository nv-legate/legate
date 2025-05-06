/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 */

#include <legate.h>

#include <iostream>
#define PYBIND11_DETAILED_ERROR_MESSAGES
#include <cstdint>
#include <pybind11/pybind11.h>
#include <type_traits>

namespace hello_world {

class HelloWorld : public legate::LegateTask<HelloWorld> {
 public:
  static inline const auto TASK_CONFIG =  // NOLINT(cert-err58-cpp)
    legate::TaskConfig{legate::LocalTaskID{0}};

  static void cpu_variant(legate::TaskContext);
};

void HelloWorld::cpu_variant(legate::TaskContext) { std::cout << "Hello World!\n"; }

}  // namespace hello_world

namespace {

template <typename T>
constexpr std::underlying_type_t<T> to_underlying(T e)
{
  static_assert(std::is_enum_v<T>);
  return static_cast<std::underlying_type_t<T>>(e);
}

}  // namespace

namespace py = pybind11;

PYBIND11_MODULE(hello_world_pybind11, m)
{
  m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: hello_world_pybind11

        .. autosummary::
           :toctree: _generate

           HelloWorld
    )pbdoc";

  py::class_<hello_world::HelloWorld>(m, "HelloWorld")
    .def(py::init<>())
    .def_property_readonly_static(
      "TASK_ID",
      [](const py::object& /* self */) {
        return to_underlying(hello_world::HelloWorld::TASK_CONFIG.task_id());
      })
    .def_static("register_variants", [](std::uintptr_t lib_ptr) {
      hello_world::HelloWorld::register_variants(
        *reinterpret_cast<legate::Library*>(lib_ptr)  // NOLINT(performance-no-int-to-ptr)
      );
    });

#ifdef VERSION_INFO
  m.attr("__version__") = LEGATE_STRINGIZE(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
