#ifndef CSCT_CUH_
#define CSCT_CUH_

#include<utility>
#include<stdio.h>

#include "util/format.hpp"

#define BLOCK_SIZE 16

typedef std::pair<bool*, size_t> CSCTResult;

__device__ size_t indexInImage(size_t width, size_t height, size_t radius) {
    int xCoordInImg = blockDim.x * blockIdx.x + threadIdx.x - radius;
    int yCoordInImg = blockDim.y * blockIdx.y + threadIdx.y - radius;

    /*
    If a coordinate lays outside of the image's bounds then the closest pixel is used for that value.
    This allows us to have a radius around the selected tile even for pixels on the edges of the image.
    */
    size_t xCoordClampedInImg = min(static_cast<size_t>(max(0, xCoordInImg)), width-1);
    size_t yCoordClampedInImg = min(static_cast<size_t>(max(0, yCoordInImg)), height-1);

    return yCoordClampedInImg * width + xCoordClampedInImg;
}

__device__ void saveImageBlockToSharedMemory(Byte* image, Byte imageBlock[], size_t imageIndex) {
    size_t blockIdx = blockDim.x * threadIdx.y + threadIdx.x;
    imageBlock[blockIdx] = image[imageIndex];
}

__device__ bool insideHalo(size_t radius) {
    if(threadIdx.x < radius || blockDim.x - radius <= threadIdx.x) return false;
    else if(threadIdx.y < radius || blockDim.y - radius <= threadIdx.y) return false;
    return true;
}

__device__ void censusTransform(Byte* imageBlock, bool* results, size_t imageIndex, size_t radius) {
    size_t diameter = 2 * radius + 1;
    size_t resultSize = ((diameter * (diameter + 1)) / 2) - 2;

    size_t resIdx = 0;
    int radiusInt = static_cast<int>(radius);
    for(int diffY = -radiusInt; diffY <= radiusInt; ++diffY) {
        for(int diffX = -radiusInt; diffX <= diffY; ++diffX) {
            if(diffY == 0 && diffX == 0) continue;
            else if(diffY == radius && diffX == radiusInt) continue;
            size_t diffIdx = (threadIdx.y + diffY) * blockDim.x + (threadIdx.x + diffX);
            size_t oppIdx = (threadIdx.y - diffY) * blockDim.x + (threadIdx.x - diffX);

            results[(resultSize * imageIndex) + resIdx] = imageBlock[diffIdx] >= imageBlock[oppIdx];
            ++resIdx;
        }
    }
}

__device__ inline void csct(Byte* image, bool* results, size_t width, size_t height, size_t radius) {
    __shared__ Byte imageBlock[BLOCK_SIZE * BLOCK_SIZE];

    size_t imageIndex = indexInImage(width, height, radius);
    saveImageBlockToSharedMemory(image, imageBlock, imageIndex);
    __syncthreads();

    if(insideHalo(radius))
        censusTransform(imageBlock, results, imageIndex, radius);
};

#endif