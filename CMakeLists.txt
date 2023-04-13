#-----------------------------------------------------------------------
# - Top Level CMakeLists.txt for AltMin Build
#-----------------------------------------------------------------------
# - Enforce an out-of-source builds before anything else
#
if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_BINARY_DIR})
  message(STATUS "AltMin requires an out-of-source build.")
  message(STATUS "Please remove these files from ${CMAKE_CURRENT_BINARY_DIR} first:")
  message(STATUS "CMakeCache.txt")
  message(STATUS "CMakeFiles")
  message(STATUS "Once these files are removed, create a separate directory")
  message(STATUS "and run CMake from there")
  message(FATAL_ERROR "in-source build detected")
endif()

project(atlmin LANGUAGES CXX)
cmake_minimum_required(VERSION 3.18...3.22)
find_package(Python 3.7 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)
# set(CMAKE_POLICY_DEFAULT_CMP0097 NEW)


if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# using the highest C++ standard supported by the compiler
# at least c++17 is required
set(ALTMIN_CXX_STANDARD DETECT CACHE STRING "Override the default CXX_STANDARD to compile with.")
set_property(CACHE ALTMIN_CXX_STANDARD PROPERTY STRINGS DETECT 23 20 17)

if(ALTMIN_CXX_STANDARD STREQUAL "DETECT")
foreach(CXX_STANDARD_VAR 23;20;17)
  if("cxx_std_${CXX_STANDARD_VAR}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
      message(STATUS "Detected support for C++${CXX_STANDARD_VAR} standard")
      set(CMAKE_CXX_STANDARD ${CXX_STANDARD_VAR})
      break()
  endif()
endforeach()
endif()

# version of the libraries to be used
set(CPM_DOWNLOAD_VERSION 0.38.1)
set(EIGEN_DOWNLOAD_VERSION 3.4.0)
set(NANOBIND_DOWNLOAD_VERSION v1.1.1)
set(CATCH2_DOWNLOAD_VERSION v3.3.2)

option(ALTMIN_BUILD_TESTS "Build tests" ON)

if (CPM_SOURCE_CACHE)
    set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
elseif (DEFINED ENV{CPM_SOURCE_CACHE})
    set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
else ()
    set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
endif ()
if (NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
            https://github.com/TheLartians/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
            ${CPM_DOWNLOAD_LOCATION}
            )
endif ()
include(${CPM_DOWNLOAD_LOCATION})

CPMAddPackage(
        NAME eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen
        GIT_TAG ${EIGEN_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM YES
)

CPMAddPackage(
        NAME nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind
        GIT_TAG ${NANOBIND_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM YES
)

# add_subdirectory(${nanobind_SOURCE_DIR} ${nanobind_BINARY_DIR} EXCLUDE_FROM_ALL)

add_executable(eigen_example src/eigen.cpp)
target_link_libraries(eigen_example PRIVATE Eigen3::Eigen)

nanobind_add_module(nanobind_hello_world src/nanobind_hello_world.cpp)

nanobind_add_module(nanobind_hello_world_in src/nanobind_hello_world_in.cpp)

nanobind_add_module(nanobind_hello_world_out src/nanobind_hello_world_out.cpp)

nanobind_add_module(nanobind_matrix_multiplication src/nanobind_matrix_multiplication.cpp)
target_link_libraries(nanobind_matrix_multiplication PRIVATE Eigen3::Eigen)

nanobind_add_module(nanobind_return_matrix src/nanobind_return_matrix.cpp)
target_link_libraries(nanobind_return_matrix PRIVATE Eigen3::Eigen)

nanobind_add_module(nanobind_matrix_in src/nanobind_matrix_in.cpp)
target_link_libraries(nanobind_matrix_in PRIVATE Eigen3::Eigen)

nanobind_add_module(nanobind_lin src/nanobind_lin.cpp)
target_link_libraries(nanobind_lin PRIVATE Eigen3::Eigen)

nanobind_add_module(nanobind_relu src/nanobind_relu.cpp)
target_link_libraries(nanobind_relu PRIVATE Eigen3::Eigen)

install(TARGETS eigen_example DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(TARGETS nanobind_hello_world DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(TARGETS nanobind_hello_world_in DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(TARGETS nanobind_hello_world_out DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(TARGETS nanobind_matrix_multiplication DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(TARGETS nanobind_return_matrix DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(TARGETS nanobind_matrix_in DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(TARGETS nanobind_lin DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(TARGETS nanobind_relu DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)

install(FILES src/nanobind_hello_world.py DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)
install(FILES tests/tests.py DESTINATION ${CMAKE_SOURCE_DIR}/artifacts )

if(ALTMIN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
