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

SET(SOURCE_CXX_FILES src/main.cpp src/image.cpp src/sgbm.cpp)
SET(SOURCE_CUDA_FILES src/csct.cu src/hamming.cu src/directional.cu src/median.cu)

add_executable(${PROJECT_NAME} ${SOURCE_CXX_FILES} ${SOURCE_CUDA_FILES})

target_include_directories(${PROJECT_NAME} PRIVATE include lib/include)