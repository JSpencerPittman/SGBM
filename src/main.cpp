#include <iostream>
#include <stdio.h>
#include <filesystem>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "image.h"
#include "csct.cuh"
#include "hamming.cuh"
#include "directional.cuh"

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
    
    /* ROI - Original */
    // printf("ROI - Original\nROI");
    // for(size_t col = 64; col <= 69; ++col) {
    //         printf("%5lu ", col);
    // }
    // printf("\n\n");
    // for(size_t row = 97; row <= 101; ++row) {
    //     printf("%3lu", row);
    //     for(size_t col = 64; col <= 69; ++col) {
    //         printf("%5d ", leftImage.data()[row * imageWidth + col]);
    //     }
    //     printf("\n");
    // }
    /* -------------- */

    // Center-Symmetric Census Transform
    CSCTResults leftCSCTRes = csct(leftImage);
    CSCTResults rightCSCTRes = csct(rightImage);

    /* ROI - CSCT */
    // printf("\nROI - CSCT\nROI");
    // for(size_t col = 64; col <= 69; ++col) {
    //         printf("%15lu", col);
    // }
    // printf("\n\n");
    // for(size_t row = 97; row <= 101; ++row) {
    //     printf("%3lu", row);
    //     for(size_t col = 64; col <= 69; ++col) {
    //         printf("  ");
    //         for(size_t comp = 0; comp < leftCSCTRes.compPerPix; ++comp) {
    //             size_t compIdx = (row * imageWidth + col) * leftCSCTRes.compPerPix + comp;
    //             printf("%d", leftCSCTRes.data[compIdx]);
    //         }
    //         printf(" ");
    //     }
    //     printf("\n");
    // }
    /* -------------- */

    printf("Completed Census Transforms!\n");

    HamDistances hams = hamming(leftCSCTRes, rightCSCTRes, imageWidth, imageHeight);

    delete [] leftCSCTRes.data;
    delete [] rightCSCTRes.data;

    printf("Completed Hamming Distance Calculations!\n");

    size_t croppedImageWidth = hams.numPixels / imageHeight;
    uint8_t* disparityMap = directionalLoss(hams, croppedImageWidth, imageHeight);

    /* ROI - Disparity Map */
    // printf("\nROI - Disparity Map\nROI");
    // for(size_t col = 195; col <= 205; ++col) {
    //         printf("%6lu", col);
    // }
    // printf("\n\n");
    // for(size_t row = 195; row <= 205; ++row) {
    //     printf("%3lu ", row);
    //     for(size_t col = 131; col <= 141; ++col) {
    //         printf("%5d ", disparityMap[row * croppedImageWidth + col]);
    //     }
    //     printf("\n");
    // }
    /* -------------- */


    delete [] hams.data;

    printf("Finished Estimating Disparity Map!\n");

    std::filesystem::path outPath("disparity.png");
    stbi_write_png(outPath.c_str(), croppedImageWidth, imageHeight, 1, disparityMap, croppedImageWidth);

    delete [] disparityMap;

    return 0;
}