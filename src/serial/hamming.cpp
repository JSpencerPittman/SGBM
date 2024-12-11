#include "serial/sgbm.h"

uint32_t calcDistance(bool *bitSeq1, bool *bitSeq2, size_t seqLen)
{
    uint32_t distance = 0;
    for (size_t idx = 0; idx < seqLen; ++idx)
        if (bitSeq1[idx] != bitSeq2[idx])
            distance++;
    return distance;
}


void hamming(Tensor<bool>& leftCSCT, Tensor<bool>& rightCSCT, Tensor<uint32_t>& distances)
{
    size_t width = leftCSCT.dims.cols;
    size_t height = leftCSCT.dims.rows;
    size_t compPerPixel = leftCSCT.dims.channels;
    size_t croppedWidth = width - MAX_DISPARITY;

    for(size_t row = 0; row < leftCSCT.dims.rows; ++row) {
        for(size_t col = 0; col < croppedWidth; ++col) {
            TensorCoord coordCrop(row, col);
            TensorCoord coordLeftImage(coordCrop.row, coordCrop.col + MAX_DISPARITY);
            for(size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity) {
                TensorCoord coordRightImage(coordLeftImage.row, coordLeftImage.col - disparity);
                if(row == 20 && col == 20 && disparity == 0) {
                    printf("HAMMING TIME\n");
                    uint32_t dists = calcDistance(leftCSCT.colPtr(coordLeftImage),
                                rightCSCT.colPtr(coordRightImage),
                                compPerPixel);
                    printf("Distance: %u\nLeft: ", dists);
                    for(size_t idx = 0; idx < compPerPixel; ++idx) {
                        printf("%d, ", leftCSCT.colPtr(coordLeftImage)[idx]);
                    }
                    printf("\nRight: ");
                    for(size_t idx = 0; idx < compPerPixel; ++idx) {
                        printf("%d, ", rightCSCT.colPtr(coordRightImage)[idx]);
                    }
                    printf("\n");
                }
                distances(coordCrop.row, coordCrop.col, disparity) = 
                    calcDistance(leftCSCT.colPtr(coordLeftImage),
                                rightCSCT.colPtr(coordRightImage),
                                compPerPixel);
            }
        }
    }
}

Tensor<uint32_t> allocateDistancesArray(size_t width, size_t height)
{
    TensorDims distShape (height, width - MAX_DISPARITY, MAX_DISPARITY+1);
    return {distShape};
}


Tensor<uint32_t> hamming(Tensor<bool>& leftCSCT, Tensor<bool>& rightCSCT)
{
    size_t width = leftCSCT.dims.cols;
    size_t height = leftCSCT.dims.rows;

    Tensor<uint32_t> distances = allocateDistancesArray(width, height);

    hamming(leftCSCT, rightCSCT, distances);

    return distances;
}