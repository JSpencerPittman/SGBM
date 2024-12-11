#include "serial/sgbm.h"

Image sgbm(Image& leftImage, Image& rightImage) {
    // Center-Symmetric Census Transform
    Tensor<bool> leftCSCTRes = csct(leftImage);
    Tensor<bool> rightCSCTRes = csct(rightImage);

    // Hamming Distance Calculation
    Tensor<uint32_t> hams = hamming(leftCSCTRes, rightCSCTRes);
    leftCSCTRes.free();
    rightCSCTRes.free();

    // Directional Loss
    Tensor<Byte> messyDisparityMap = directionalLoss(hams);
    hams.free();

    // Median Blurring
    Tensor<Byte> disparityMap = rankFilter(messyDisparityMap, RankOperation::Median);
    messyDisparityMap.free();

    // Dilation & Erosion Loop
    if(MORPH_ITERATIONS > 0) {
        for(size_t idx = 0; idx < MORPH_ITERATIONS; ++idx) {
            Tensor<Byte> nextDisparityMap = rankFilter(disparityMap, RankOperation::Maximum);
            disparityMap.free();
            disparityMap = nextDisparityMap;
        }

        for(size_t idx = 0; idx < MORPH_ITERATIONS; ++idx) {
            Tensor<Byte> nextDisparityMap = rankFilter(disparityMap, RankOperation::Minimum);
            disparityMap.free();
            disparityMap = nextDisparityMap;
        }
    }

    return Image(disparityMap);
}