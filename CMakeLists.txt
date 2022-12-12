cmake_minimum_required(VERSION 3.21)
project(CUDA3.2 LANGUAGES CXX CUDA)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -fopenmp")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

set(CMAKE_CXX_STANDARD 14)

add_executable(CUDA3.2 main.cu)

target_compile_options(CUDA3.2 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -Xcompiler=-fopenmp>)

set_target_properties(CUDA3.2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)


set_target_properties(CUDA3.2 PROPERTIES CUDA_ARCHITECTURES "52;61;75")
