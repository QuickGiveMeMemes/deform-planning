cmake_minimum_required(VERSION 3.18)


project(diffadmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# DiffADMM
set(DIFFADMM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/diff_ADMM")
set(DIFFADMM_INC  "${DIFFADMM_ROOT}/include")
set(DIFFADMM_SRC  "${DIFFADMM_ROOT}/src")

# LEAP
set(LEAP_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/leap-discrete/refactored_codebase_v1")
set(LEAP_INC  "${LEAP_ROOT}/include")
set(LEAP_SRC  "${LEAP_ROOT}/src")

# Integration
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")
set(PROJECT_INC  "${PROJECT_ROOT}/include")
set(PROJECT_SRC  "${PROJECT_ROOT}/src/cpp")

find_package(Kokkos REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)
find_package(pinocchio CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)
find_library(CPPAD_LIBRARY NAMES cppad_lib cppad HINTS $ENV{CONDA_PREFIX}/lib /opt/homebrew/lib)
find_path(CPPAD_INCLUDE_DIR cppad/cppad.hpp HINTS $ENV{CONDA_PREFIX}/include /opt/homebrew/include)

find_library(CHOLMOD_LIBRARY NAMES cholmod HINTS $ENV{CONDA_PREFIX}/lib)
find_path(CHOLMOD_INCLUDE_DIR cholmod.h
HINTS $ENV{CONDA_PREFIX}/include
PATH_SUFFIXES suitesparse)

file(GLOB_RECURSE LEAP_SOURCES CONFIGURE_DEPENDS ${LEAP_SRC}/*.cpp)
list(FILTER LEAP_SOURCES EXCLUDE REGEX ".*/(apps|examples/apps|bin)/.*")

add_library(leap STATIC ${LEAP_SOURCES})
target_include_directories(leap PUBLIC 
  ${LEAP_INC}
  ${CPPAD_INCLUDE_DIR})
target_link_libraries(leap PUBLIC 
 Eigen3::Eigen 
 pinocchio::pinocchio
 ${CPPAD_LIBRARY})
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(leap PUBLIC OpenMP::OpenMP_CXX)   # kkt_assembler's #pragma omp
endif()



if(CHOLMOD_LIBRARY AND CHOLMOD_INCLUDE_DIR)
  target_include_directories(leap PUBLIC ${CHOLMOD_INCLUDE_DIR})
  target_link_libraries(leap PUBLIC ${CHOLMOD_LIBRARY})
  target_compile_definitions(leap PUBLIC LEAP_HAS_CHOLMOD)
  message(STATUS "CHOLMOD found: ${CHOLMOD_LIBRARY}")
else()
  message(STATUS "CHOLMOD not found; cholmod backend disabled")
endif()

add_executable(rope_residual 
  ${PROJECT_SRC}/rope_residual_opt.cpp
  ${PROJECT_SRC}/rope_model.cpp)

target_include_directories(rope_residual PRIVATE
  ${PROJECT_INC}
  ${DIFFADMM_INC})

target_link_libraries(rope_residual PRIVATE leap yaml-cpp)     

# This breaks everything
# include(CheckCXXCompilerFlag)
# check_cxx_compiler_flag("-march=native" HAS_MARCH_NATIVE)
# if(HAS_MARCH_NATIVE)
#   target_compile_options(leap PUBLIC -march=native)
#   target_compile_options(rope_residual PUBLIC -march=native)
# endif()

# target_compile_options(rope_residual PRIVATE -fno-omit-frame-pointer -g)
# target_link_options(rope_residual PRIVATE -no-pie)

# target_compile_options(leap PUBLIC -O3 -ffast-math)
# target_compile_options(rope_residual PRIVATE -O3 -ffast-math)
