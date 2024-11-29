#include <iostream>
#include <stdio.h>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"
#include "csct.cuh"
#include "hamming.cuh"
#include "directional.cuh"
#include "tensor.cuh"
#include "median.cuh"

int main()
{
    std::filesystem::path leftImagePath = "../data/left.png";
    std::filesystem::path rightImagePath = "../data/right.png";

    Image leftImage(leftImagePath, true);
    Image rightImage(rightImagePath, true);
    printf("Loaded left and right image.\n");

    // Center-Symmetric Census Transform
    FlatImage leftImageF(leftImage.data()->dims, leftImage.data()->data);
    FlatImage rightImageF(rightImage.data()->dims, rightImage.data()->data);
    CSCTResults leftCSCTRes = csct(leftImageF);
    CSCTResults rightCSCTRes = csct(rightImageF);
    printf("Completed Center-Symmetric Census Transform.\n");

    // Hamming Distance Calculation
    Distances hams = hamming(leftCSCTRes, rightCSCTRes);
    leftCSCTRes.free();
    rightCSCTRes.free();
    printf("Completed Hamming Distance Calculations.\n");

    // Directional Loss
    FlatImage disparityMap = directionalLoss(hams);
    hams.free();
    printf("Completed Directional Loss.\n");

    // Median Blurring
    FlatImage blurDispMap = medianBlur(disparityMap);
    disparityMap.free();
    printf("Completed Median Blurring.\n");

    // Save Disparity Map
    size_t imageWidth = leftImage.width();
    size_t imageHeight = rightImage.height();
    std::filesystem::path outPath("disparity_blur.png");
    stbi_write_png(outPath.c_str(), hams.dims.cols, imageHeight, 1, blurDispMap.data, hams.dims.cols);
    blurDispMap.free();
    printf("Saved Disparity Map To Disk.\n");

    return 0;
}