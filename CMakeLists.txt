cmake_minimum_required(VERSION 3.2)
project(matrix_mul VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_FLAGS "-std=c++11 -fPIC -fpic -Wall -Wextra -pedantic -O3")

add_subdirectory(thirdparty/av)
add_subdirectory(src)


