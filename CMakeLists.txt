cmake_minimum_required(VERSION 3.8)
project(hello_cuda VERSION 0.1.0 LANGUAGES C CXX CUDA)



add_definitions(-std=c++11)

file(GLOB CU "src/cuda/*.cu")
file(GLOB CPP "src/cpp/*.cpp")
add_executable(hello_cuda ${CU} ${CPP})


target_include_directories(
    hello_cuda
    PRIVATE
    "src/cuda/"
)
target_include_directories(
    hello_cuda
    PRIVATE 
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
target_link_libraries(
    hello_cuda
    ${CUDART_LIBRARY}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
