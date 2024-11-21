cmake_minimum_required(VERSION 3.25)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(SGBM VERSION 1.0)

add_executable(${PROJECT_NAME} src/main.cpp)
