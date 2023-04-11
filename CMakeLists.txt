project(c++)
cmake_minimum_required(VERSION 3.18...3.22)
find_package(Python 3.7 COMPONENTS Interpreter Development.Module REQUIRED)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Detect the installed nanobind package and import it into CMake
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)
find_package(Eigen3 3.4 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
add_executable(eigen src/eigen.cpp)
target_link_libraries (eigen Eigen3::Eigen)

nanobind_add_module(nanobind_hello_world src/nanobind_hello_world.cpp)
nanobind_add_module(nanobind_hello_world_in src/nanobind_hello_world_in.cpp)
nanobind_add_module(nanobind_hello_world_out src/nanobind_hello_world_out.cpp)
nanobind_add_module(nanobind_matrix_multiplication src/nanobind_matrix_multiplication.cpp)
nanobind_add_module(nanobind_return_matrix src/nanobind_return_matrix.cpp)
nanobind_add_module(nanobind_matrix_in src/nanobind_matrix_in.cpp)