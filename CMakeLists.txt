cmake_minimum_required(VERSION 3.25)

# C++ version
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# CUDA C++ version
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Add C++ and CUDA to used languages
enable_language(CXX)
enable_language(CUDA)

project(SGBM VERSION 1.0)

add_executable(${PROJECT_NAME} src/main.cpp)
add_executable("${PROJECT_NAME}_CUDA" src/hello_world.cu)
