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

find_package(pybind11 CONFIG REQUIRED)
find_package(Kokkos REQUIRED)
find_package(Eigen3 REQUIRED CONFIG)
find_package(pinocchio CONFIG QUIET)


# pybind11_add_module(diffadmm
#   ${CMAKE_CURRENT_SOURCE_DIR}/diffadmm_pybind.cpp
#   ${CMAKE_CURRENT_SOURCE_DIR}/diffadmm_wrapper.cpp
#   ${DIFFADMM_SRC}/admm_forward.cpp
#   ${DIFFADMM_SRC}/admm_backward.cpp
#   ${DIFFADMM_SRC}/admm_utils.cpp)

# target_include_directories(diffadmm PRIVATE
#   ${CMAKE_CURRENT_SOURCE_DIR}        
#   ${DIFFADMM_INC})   

# target_link_libraries(diffadmm PRIVATE 
#   pybind11::module
#   Kokkos::kokkos
# )

# For Linting (temporary)
add_library( dummy OBJECT
    ${PROJECT_SRC}/dummy.cpp
)

target_include_directories(dummy PRIVATE
    ${PROJECT_INC}
    ${LEAP_INC}
    ${DIFFADMM_INC}
)

target_link_libraries(dummy PRIVATE 
  pybind11::module
  Kokkos::kokkos
  Eigen3::Eigen
)