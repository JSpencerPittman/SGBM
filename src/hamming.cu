#include "hamming.cuh"

#include <stdio.h>

__device__ uint32_t calcDistance(bool *bitSeq1, bool *bitSeq2, size_t seqLen)
{
    uint32_t distance = 0;
    for (size_t idx = 0; idx < seqLen; ++idx)
        if (bitSeq1[idx] == bitSeq2[idx])
            distance++;
    return distance;
}

__global__ void hammingKernel(bool *leftCSCT, bool *rightCSCT, uint32_t *distances,
                              size_t width, size_t height, size_t compPerPixel)
{  
    size_t croppedWidth = width - MAX_DISPARITY;
    size_t yCoord = blockIdx.y * blockDim.y + threadIdx.y;
    size_t xCoordCrop = blockIdx.x * blockDim.x + threadIdx.x;
    size_t xCoordLeftImage = xCoordCrop + MAX_DISPARITY;

    if(xCoordCrop >= croppedWidth) return;

    size_t imageIndexLeft = yCoord * width + xCoordLeftImage;
    size_t pixelIdx = yCoord * croppedWidth + xCoordCrop;
    size_t distancesIdx = pixelIdx * (MAX_DISPARITY + 1);
    bool* leftPixel = leftCSCT + compPerPixel * imageIndexLeft;

    for(size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity) {
        size_t imageIndexInRight = imageIndexLeft - disparity;
        bool* rightPixel = rightCSCT + compPerPixel * imageIndexInRight;
        distances[distancesIdx + disparity] = calcDistance(leftPixel, rightPixel, compPerPixel);
    }
}

CSCTResults copyResultsToDevice(CSCTResults resultsHost)
{
    CSCTResults resultsDev(nullptr, resultsHost.numPixels, resultsHost.compPerPix);
    cudaMalloc(&resultsDev.data, resultsDev.numBytes);
    cudaMemcpy(resultsDev.data, resultsHost.data, resultsHost.numBytes, cudaMemcpyHostToDevice);
    return resultsDev;
}

HamDistances allocateHamDistancesArray(size_t width, size_t height)
{
    uint32_t *distancesDev;
    size_t numPixels = (width - MAX_DISPARITY) * height;
    size_t numBytes = numPixels * (MAX_DISPARITY+1) * sizeof(uint32_t);
    cudaMalloc(&distancesDev, numBytes);
    return {distancesDev, numPixels, MAX_DISPARITY};
}

HamDistances copyDistancesToHost(HamDistances distancesDev) {
    HamDistances distancesHost(new uint32_t[distancesDev.numBytes],
                               distancesDev.numPixels,
                               distancesDev.maxDisparity);
    cudaMemcpy(distancesHost.data, distancesDev.data, distancesDev.numBytes, cudaMemcpyDeviceToHost);
    return distancesHost;
}


HamDistances hamming(CSCTResults leftCSCT, CSCTResults rightCSCT, size_t width, size_t height)
{
    CSCTResults leftCSCTDev = copyResultsToDevice(leftCSCT);
    CSCTResults rightCSCTDev = copyResultsToDevice(rightCSCT);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(height) / BLOCK_SIZE));
    size_t blocksHorz = static_cast<size_t>(
        ceil((static_cast<float>(width) - MAX_DISPARITY) / BLOCK_SIZE));
    dim3 numBlocks(blocksHorz, blocksVert);

    HamDistances distancesDev = allocateHamDistancesArray(width, height);

    hammingKernel<<<numBlocks, threadsPerBlock>>>(leftCSCTDev.data, rightCSCTDev.data, distancesDev.data, width, height, leftCSCTDev.compPerPix);

    HamDistances distancesHost = copyDistancesToHost(distancesDev);

    cudaFree(leftCSCTDev.data);
    cudaFree(rightCSCTDev.data);
    cudaFree(distancesDev.data);

    return distancesHost;
}