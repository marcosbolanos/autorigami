cmake_minimum_required(VERSION 3.18)

project(autorigami LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(AUTORIGAMI_DEV_LINT "Run clang-tidy while compiling native targets" OFF)

enable_testing()

add_subdirectory(cpp)
