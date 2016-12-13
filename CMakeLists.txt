cmake_minimum_required(VERSION 3.6)
project(nn_cuda)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(nn_cuda ${SOURCE_FILES})