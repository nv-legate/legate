/* Copyright 2021 NVIDIA Corporation
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

#pragma once

#include "legion.h"

#include "utilities/typedefs.h"

namespace legate {

extern uint32_t extract_env(const char* env_name,
                            const uint32_t default_value,
                            const uint32_t test_value);

class Core {
 public:
  static void parse_config(void);
  static void shutdown(void);

 public:
  // Configuration settings
  static bool show_progress;
#ifdef LEGATE_USE_CUDA
 public:
  static cublasContext* get_cublas(void);
#endif
};

class ResourceConfig;
class LibraryContext;

class Runtime {
 public:
  using MainFnPtr = void (*)(int32_t, char**, Runtime*);

 public:
  Runtime();
  ~Runtime();

 public:
  friend void initialize(int32_t argc, char** argv);
  friend void set_main_function(MainFnPtr p_main);
  friend int32_t start(int32_t argc, char** argv);

 public:
  void set_main_function(MainFnPtr main_fn);
  MainFnPtr get_main_function() const;

 public:
  LibraryContext* find_library(const std::string& library_name, bool can_fail = false) const;
  LibraryContext* create_library(const std::string& library_name, const ResourceConfig& config);

 public:
  static void initialize(int32_t argc, char** argv);
  static int32_t start(int32_t argc, char** argv);
  static Runtime* get_runtime();

 private:
  static Runtime* runtime_;

 private:
  std::map<std::string, LibraryContext*> libraries_;

 private:
  MainFnPtr main_fn_{nullptr};
};

void initialize(int32_t argc, char** argv);

void set_main_function(Runtime::MainFnPtr p_main);

int32_t start(int32_t argc, char** argv);

}  // namespace legate
