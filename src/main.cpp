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

int main() {
    // Load Images
    std::filesystem::path leftImagePath = "../data/left.png";
    std::filesystem::path rightImagePath = "../data/right.png";

    Image leftImage(leftImagePath, true);
    Image rightImage(rightImagePath, true);

    // Save Grayscale images
    std::filesystem::path leftImageGraySavePath = "../data/left_gray.png";
    std::filesystem::path rightImageGraySavePath = "../data/right_gray.png";
    leftImage.writePng(leftImageGraySavePath);
    rightImage.writePng(rightImageGraySavePath);

    size_t imageWidth = leftImage.width();
    size_t imageHeight = rightImage.height();

    // Center-Symmetric Census Transform
    TensorDims imageDims(leftImage.height(), leftImage.width(), leftImage.channels());
    FlatImage leftImageF(imageDims, leftImage.data());
    FlatImage rightImageF(imageDims, rightImage.data());

    CSCTResults leftCSCTRes = csct(leftImageF);
    CSCTResults rightCSCTRes = csct(rightImageF);

    printf("Completed Census Transforms!\n");

    Distances hams = hamming(leftCSCTRes, rightCSCTRes);

    leftCSCTRes.free();
    rightCSCTRes.free();

    printf("Completed Hamming Distance Calculations!\n");

    FlatImage disparityMap = directionalLoss(hams);

    hams.free();

    printf("Finished Estimating Disparity Map!\n");

    // std::filesystem::path outPath("disparity.png");
    // stbi_write_png(outPath.c_str(), hams.dims.cols, imageHeight, 1, disparityMap.data, hams.dims.cols);

    FlatImage blurDispMap = medianBlur(disparityMap);

    disparityMap.free();

    printf("Finished Blurring Disparity Map!\n");

    std::filesystem::path outPath2("disparity_blur.png");
    stbi_write_png(outPath2.c_str(), hams.dims.cols, imageHeight, 1, blurDispMap.data, hams.dims.cols);

    blurDispMap.free();

    return 0;
}