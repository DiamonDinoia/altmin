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

option(ALTMIN_ENABLE_LTO "Enables link time optimisation" ON)

cmake_minimum_required(VERSION 3.18...3.26)
project(fast_altmin LANGUAGES CXX)

find_package(Python 3.7 COMPONENTS Interpreter Development.Module REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(DCMAKE_RULE_MESSAGES OFF)


if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()


# using the highest C++ standard supported by the compiler
# at least c++17 is required
set(ALTMIN_CXX_STANDARD DETECT CACHE STRING "Override the default CXX_STANDARD to compile with.")
set_property(CACHE ALTMIN_CXX_STANDARD PROPERTY STRINGS DETECT 23 20)

if(ALTMIN_CXX_STANDARD STREQUAL "DETECT")
foreach(CXX_STANDARD_VAR 23;20)
  if("cxx_std_${CXX_STANDARD_VAR}" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
      message(STATUS "Detected support for C++${CXX_STANDARD_VAR} standard")
      set(ALTMIN_CXX_STANDARD ${CXX_STANDARD_VAR})
      break()
  endif()
endforeach()
endif()

set(FLAGS_DEBUG -O0 -fstack-protector-all -Wall -Wextra -pedantic -fno-inline)
set(FLAGS_RELEASE -O3 -ffp-contract=fast -freciprocal-math
        -fmerge-all-constants -fno-math-errno -march=native -funroll-loops
        -ftree-vectorize -fno-trapping-math -fassociative-math -ffinite-math-only
        -fno-signed-zeros)
set(FLAGS_RelWithDebInfo ${FLAGS_RELEASE} -g -fno-omit-frame-pointer)

if (NOT (CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    list(APPEND FLAGS_RELEASE -fcx-limited-range)
endif ()

list(APPEND FLAGS $<$<CONFIG:DEBUG>:${FLAGS_DEBUG}>
        $<$<CONFIG:Release>:${FLAGS_RELEASE}>
        $<$<CONFIG:RelWithDebInfo>:${FLAGS_RelWithDebInfo}>)

function(target_add_compilation_flags target)
    set(old_flags ${FLAGS})
    message(STATUS "COMPILATION FLAGS ${old_flags}")
    target_compile_options(${target} PUBLIC  "$<$<COMPILE_LANGUAGE:CXX>:${old_flags}>")
    if (GPU)
        if (NOT "${old_flags}" STREQUAL "")
            string(REPLACE ";" "," CUDA_flags "${old_flags}")
            string(REPLACE "-pedantic" "-Wno-pedantic" CUDA_flags "${CUDA_flags}")
            target_compile_options(${target} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${CUDA_flags}>")
        endif ()
    endif ()
endfunction()

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    if (POLICY CMP0069)
        cmake_policy(SET CMP0069 NEW)
        set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
    endif ()
    message(STATUS "IPO / LTO enabled")
else ()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif ()

function(enable_lto target)
    if (supported)
        set_target_properties(${target} PROPERTIES INTERPROCEDURAL_OPTIMIZATION ${ALTMIN_ENABLE_LTO})
    else ()
        message(WARNING "IPO / LTO not supported: <${error}>")
    endif ()
endfunction()


# version of the libraries to be used
set(CPM_DOWNLOAD_VERSION 0.38.1)
set(EIGEN_DOWNLOAD_VERSION 3.4.0)
set(NANOBIND_DOWNLOAD_VERSION v1.1.1)
#set(CATCH2_DOWNLOAD_VERSION v3.3.2)

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
        NAME Eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen
        GIT_TAG ${EIGEN_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
        DOWNLOAD_ONLY YES 

)

if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen SYSTEM INTERFACE ${Eigen_SOURCE_DIR})
endif()

CPMAddPackage(
        NAME nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind
        GIT_TAG ${NANOBIND_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM
)


nanobind_add_module(fast_altmin src/fast_altmin.cpp)
target_include_directories(fast_altmin PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fast_altmin PRIVATE Eigen)

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
    list(APPEND FLAGS ${OpenMP_CXX_FLAGS})
    target_link_libraries(fast_altmin PRIVATE OpenMP::OpenMP_CXX)
endif()

target_add_compilation_flags(fast_altmin)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    enable_lto(fast_altmin)
endif()

install(TARGETS fast_altmin LIBRARY DESTINATION .)

if(ALTMIN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
