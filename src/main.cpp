#include <stdio.h>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"
#include "sgbm.h"

int main()
{
    std::filesystem::path leftImagePath = "../data/left.png";
    std::filesystem::path rightImagePath = "../data/right.png";

    Image leftImage(leftImagePath, true);
    Image rightImage(rightImagePath, true);
    printf("Loaded left and right image.\n");

    Image blurDispMap = sgbm(leftImage, rightImage);

    std::filesystem::path outPath("disparity_blur.png");
    blurDispMap.writePng(outPath);
    printf("Saved Disparity Map To Disk.\n");

    return 0;
}