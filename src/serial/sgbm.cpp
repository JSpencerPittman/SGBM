#include "serial/sgbm.h"

Image sgbm(Image& leftImage, Image& rightImage) {
    // Center-Symmetric Census Transform
    Tensor<bool> leftCSCTRes = csct(leftImage);
    Tensor<bool> rightCSCTRes = csct(rightImage);

    // Hamming Distance Calculation
    Tensor<uint32_t> hams = hamming(leftCSCTRes, rightCSCTRes);
    leftCSCTRes.free();
    rightCSCTRes.free();

    printf("HAMMING TEST\n");
    for(size_t idx = 0; idx <= MAX_DISPARITY; ++idx)
        printf("%u, ", hams(10, 20, idx));
    printf("\n");

    // Directional Loss
    Tensor<Byte> messyDisparityMap = directionalLoss(hams);
    hams.free();

    return {messyDisparityMap};
}