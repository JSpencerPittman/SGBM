#include <stdio.h>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "serial/image.h"
#include "serial/sgbm.h"

#include <chrono>

int main(int argc, char* argv[])
{
    if(argc != 3 && argc != 4)
        throw std::runtime_error("Missing arguments: path to left and right image paths, respectively.");

    std::filesystem::path leftImagePath(argv[1]);
    std::filesystem::path rightImagePath(argv[2]);
    
    std::filesystem::path outputPath;
    if(argc == 4) outputPath = std::filesystem::path(argv[3]);
    else outputPath = "disparity.png";

    Image leftImage(leftImagePath, true);
    Image rightImage(rightImagePath, true);

    auto start = std::chrono::system_clock::now();

    // Image blurDispMap = sgbm(leftImage, rightImage);
    sgbm(leftImage, rightImage);

    auto end = std::chrono::system_clock::now();
    auto duration = std::chrono::duration<double>(end - start);

    // blurDispMap.writePng(outputPath);

    printf("Duration: %lf\n", duration.count());

    return 0;
}