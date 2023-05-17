project(atlmin LANGUAGES CXX)
cmake_minimum_required(VERSION 3.18...3.26)

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

# version of the libraries to be used
set(CPM_DOWNLOAD_VERSION 0.38.1)
set(EIGEN_DOWNLOAD_VERSION 3.4.0)
set(NANOBIND_DOWNLOAD_VERSION v1.1.1)
set(CATCH2_DOWNLOAD_VERSION v3.3.2)
set(FLASHLIGHT_DOWNLOAD_VERSION v0.4.0)
set(ARRAYFIRE_DOWNLOAD_VERSION v3.8.3)
SET(LAPACK_DOWNLOAD_VERSION v3.11.0)
SET(SPAN_LITE_DOWNLOAD_VERSION v0.10.3)
set(FFTW_DOWNLOAD_VERSION 3.3.10)

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
        SYSTEM YES
        DOWNLOAD_ONLY YES 

)

if(Eigen_ADDED)
  add_library(Eigen INTERFACE IMPORTED)
  target_include_directories(Eigen INTERFACE ${Eigen_SOURCE_DIR})
endif()

CPMAddPackage(
        NAME gsl
        GITHUB_REPOSITORY ampl/gsl
        VERSION 2.7.0
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM 
)


CPMAddPackage(
        NAME nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind
        GIT_TAG ${NANOBIND_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM 
)

# CPMAddPackage(
#         NAME spanlite
#         GIT_REPOSITORY https://github.com/martinmoene/span-lite
#         GIT_TAG ${SPAN_LITE_DOWNLOAD_VERSION}
#         GIT_SHALLOW YES
#         GIT_PROGRESS YES
#         EXCLUDE_FROM_ALL YES
#         SYSTEM 
#         OPTIONS "SPAN_LITE_OPT_SELECT_NONSTD ON"
# )


CPMAddPackage(
        NAME FFTW3
        OPTIONS 
            "ENABLE_AVX2 ON"
            "BUILD_TESTS OFF"
            "BUILD_SHARED_LIBS OFF"
            "ENABLE_THREADS OFF"
            "ENABLE_OPENMP OFF"
        URL "http://www.fftw.org/fftw-${FFTW_DOWNLOAD_VERSION}.tar.gz"
        URL_HASH "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
        EXCLUDE_FROM_ALL YES
        GIT_SHALLOW YES
        SYSTEM
)

CPMAddPackage(
        NAME FFTW3F
        OPTIONS 
            "ENABLE_AVX2 ON"
            "BUILD_TESTS OFF"
            "BUILD_SHARED_LIBS OFF"
            "ENABLE_FLOAT ON"
            "ENABLE_THREADS OFF"
            "ENABLE_OPENMP OFF"
        URL "http://www.fftw.org/fftw-${FFTW_DOWNLOAD_VERSION}.tar.gz"
        URL_HASH "MD5=8ccbf6a5ea78a16dbc3e1306e234cc5c"
        EXCLUDE_FROM_ALL YES
        GIT_SHALLOW YES
        SYSTEM
)

# add_library(FFTW::FFTW INTERFACE IMPORTED)

# target_include_directories(FFTW::FFTW INTERFACE ${FFTW3_SOURCE_DIR}/api)
# set(FFTW_LIBRARY fftw3)
# set(FFTWF_LIBRARY fftw3f)
# set(FFTW_INCLUDE_DIR ${FFTW3_SOURCE_DIR}/api)

CPMAddPackage(
        NAME arrayfire
        GIT_REPOSITORY https://github.com/arrayfire/arrayfire
        GIT_TAG ${ARRAYFIRE_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM 
        OPTIONS 
            "AF_BUILD_EXAMPLES OFF"
            "AF_COMPUTE_LIBRARY FFTW"
            "AF_BUILD_CPU ON"
            "AF_BUILD_CUDA OFF"
            "AF_BUILD_OPENCL OFF"
            "BUILD_TESTING OFF"
)

add_library(FFTW::FFTW ALIAS fftw3)
add_library(FFTW::FFTWF ALIAS fftw3f)

# target_link_libraries(af_spdlog INTERFACE nonstd::span-lite)
# target_include_directories(af_spdlog INTERFACE ${span_lite_SOURCE_DIR}/include)

# target_link_libraries(af_glad INTERFACE span-lite)
# target_include_directories(af_glad INTERFACE ${span_lite_SOURCE_DIR}/include)

CPMAddPackage(
        NAME flashlight
        GIT_REPOSITORY https://github.com/flashlight/flashlight.git
        GIT_TAG ${FLASHLIGHT_DOWNLOAD_VERSION}
        GIT_SHALLOW YES
        GIT_PROGRESS YES
        EXCLUDE_FROM_ALL YES
        SYSTEM 
        OPTIONS 
            "ArrayFire_DIR ${arrayfire_BINARY_DIR}"
            "FL_BUILD_ARRAYFIRE ON"
            "FL_ARRAYFIRE_USE_CUDA OFF"
            "FL_ARRAYFIRE_USE_CPU ON"
            "FL_USE_ONEDNN OFF"
            "FL_BUILD_TESTS OFF"
            "FL_BUILD_EXAMPLES OFF" 
            "FL_BUILD_SCRIPTS OFF"
            "FL_BUILD_DISTRIBUTED OFF"
            "FL_BUILD_APPS OFF"
)

#install(DIRECTORY ${optim_SOURCE_DIR}/include/ DESTINATION ${CMAKE_SOURCE_DIR}/include )

nanobind_add_module(my_executable src/nanobind.cpp)
target_include_directories(my_executable PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(my_executable PRIVATE Eigen gsl flashlight)
install(TARGETS my_executable DESTINATION ${CMAKE_SOURCE_DIR}/artifacts)

if(ALTMIN_BUILD_TESTS)
    add_subdirectory(tests)
endif()
