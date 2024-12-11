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

find_package(OpenCV REQUIRED)

SET(SOURCE_CXX_FILES src/main.cpp src/image.cpp)
SET(SOURCE_CUDA_FILES src/csct.cu src/hamming.cu src/directional.cu src/rank_filt.cu src/sgbm.cu)
SET(SOURCE_CXX_FILES_SERIAL src/serial/main.cpp src/serial/csct.cpp src/serial/sgbm.cpp src/serial/image.cpp
        src/serial/hamming.cpp src/serial/directional.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_CXX_FILES} ${SOURCE_CUDA_FILES})
add_executable(OPENCV_BM benchmark/opencv_bench.cpp)
add_executable(EVAL benchmark/eval.cpp)
add_executable(SERIAL ${SOURCE_CXX_FILES_SERIAL})

target_include_directories(${PROJECT_NAME} PRIVATE include lib/include)
target_include_directories(SERIAL PRIVATE include lib/include)

target_link_libraries(OPENCV_BM PRIVATE ${OpenCV_LIBS})
target_link_libraries(EVAL PRIVATE ${OpenCV_LIBS})