cmake_minimum_required(VERSION 3.20)
project(kmsplay CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra -Werror)

option(KMSPLAY_STATIC "link the player as one static binary" OFF)
option(KMSPLAY_TESTS "build the tests" OFF)
option(KMSPLAY_COVERAGE "instrument the tests for coverage" OFF)

if(KMSPLAY_STATIC)
  set(CMAKE_FIND_LIBRARY_SUFFIXES .a)
endif()
find_package(PkgConfig REQUIRED)
pkg_check_modules(DRM REQUIRED IMPORTED_TARGET libdrm)
find_package(nlohmann_json REQUIRED)
find_package(CLI11 REQUIRED)
find_package(Threads REQUIRED)

if(KMSPLAY_COVERAGE)
  add_compile_options(--coverage -O0)
  add_link_options(--coverage)
endif()

add_library(kmsplay_core STATIC src/core.cpp src/decode.cpp src/app.cpp
                                src/kms.cpp)
target_include_directories(kmsplay_core PUBLIC src)
target_link_libraries(kmsplay_core PUBLIC PkgConfig::DRM nlohmann_json::nlohmann_json
                                          CLI11::CLI11 Threads::Threads)

add_executable(kmsplay src/main.cpp)
target_link_libraries(kmsplay PRIVATE kmsplay_core)
if(KMSPLAY_STATIC)
  target_link_options(kmsplay PRIVATE -static)
endif()
install(TARGETS kmsplay)

if(KMSPLAY_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  add_executable(kmsplay_tests tests/test_player.cpp tests/test_core.cpp
                               tests/test_decode.cpp tests/test_app.cpp)
  target_include_directories(kmsplay_tests PRIVATE tests)
  target_link_libraries(kmsplay_tests PRIVATE kmsplay_core GTest::gmock_main)
  include(GoogleTest)
  gtest_discover_tests(kmsplay_tests)
endif()
