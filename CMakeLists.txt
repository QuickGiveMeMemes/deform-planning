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

add_executable(rope_residual 
  ${PROJECT_SRC}/rope_residual_opt.cpp
  ${PROJECT_SRC}/rope_model.cpp)

target_include_directories(rope_residual PRIVATE
  ${PROJECT_INC}
  ${DIFFADMM_INC})

target_link_libraries(rope_residual PRIVATE
  leap                 # brings LEAP_INC + Eigen + pinocchio transitively
  yaml-cpp)            # or yaml-cpp::yaml-cpp on newer yaml-cpp