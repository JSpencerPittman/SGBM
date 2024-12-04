#include "sgbm.cuh"

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
    Tensor<Byte> messyDisparityMap = directionalLoss(hams);
    hams.free();
    printf("Completed Directional Loss.\n");

    // Median Blurring
    Tensor<Byte> disparityMap = rankFilter(messyDisparityMap, RankOperation::Median);
    messyDisparityMap.free();
    printf("Completed Median Blurring.\n");

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

    Tensor<Byte> disparityMapHost = disparityMap.copyToHost();
    disparityMap.free();

    return Image(disparityMapHost);
}