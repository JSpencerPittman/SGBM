#include <stdio.h>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"
#include "sgbm.h"

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
    printf("Loaded left and right image.\n");

    Image blurDispMap = sgbm(leftImage, rightImage);

    blurDispMap.writePng(outputPath);
    printf("Saved Disparity Map To Disk.\n");

    return 0;
}