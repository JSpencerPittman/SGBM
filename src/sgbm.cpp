#include "sgbm.h"

Image sgbm(Image& leftImageHost, Image& rightImageHost) {
    // Center-Symmetric Census Transform
    Tensor<bool> leftCSCTRes = csct(leftImageHost);
    Tensor<bool> rightCSCTRes = csct(rightImageHost);
    printf("Completed Center-Symmetric Census Transform.\n");

    // Hamming Distance Calculation
    Tensor<uint32_t> hams = hamming(leftCSCTRes, rightCSCTRes);
    leftCSCTRes.free();
    rightCSCTRes.free();
    printf("Completed Hamming Distance Calculations.\n");

    // Directional Loss
    Tensor<Byte> disparityMap = directionalLoss(hams);
    hams.free();
    printf("Completed Directional Loss.\n");

    // Median Blurring
    Tensor<Byte> blurDispMap = medianBlur(disparityMap);
    disparityMap.free();
    printf("Completed Median Blurring.\n");

    return Image(blurDispMap);
}