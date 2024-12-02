#include "sgbm.h"

Image sgbm(Image& leftImageHost, Image& rightImageHost) {
    // Center-Symmetric Census Transform
    CSCTResults leftCSCTRes = csct(leftImageHost);
    CSCTResults rightCSCTRes = csct(rightImageHost);
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

    return Image(blurDispMap);
}