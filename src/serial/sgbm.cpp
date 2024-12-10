#include "serial/sgbm.h"

void sgbm(Image& leftImage, Image& rightImage) {
    // Center-Symmetric Census Transform
    Tensor<bool> leftCSCTRes = csct(leftImage);
    Tensor<bool> rightCSCTRes = csct(leftImage);

    // Hamming Distance Calculation
    Tensor<uint32_t> hams = hamming(leftCSCTRes, rightCSCTRes);
    leftCSCTRes.free();
    rightCSCTRes.free();
}