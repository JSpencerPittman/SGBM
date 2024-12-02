#include "hamming.cuh"

__device__ uint32_t calcDistance(bool *bitSeq1, bool *bitSeq2, size_t seqLen)
{
    uint32_t distance = 0;
    for (size_t idx = 0; idx < seqLen; ++idx)
        if (bitSeq1[idx] != bitSeq2[idx])
            distance++;
    return distance;
}

__global__ void hammingKernel(Tensor<bool> leftCSCT, Tensor<bool> rightCSCT, Tensor<uint32_t> distances)
{
    size_t width = leftCSCT.dims.cols;
    size_t height = leftCSCT.dims.rows;
    size_t compPerPixel = leftCSCT.dims.channels;

    TensorCoord coordCrop(blockIdx.y * blockDim.y + threadIdx.y,
                          blockIdx.x * blockDim.x + threadIdx.x);
    size_t croppedWidth = width - MAX_DISPARITY;

    if(coordCrop.col >= croppedWidth || coordCrop.row >= height) return;

    TensorCoord coordLeftImage(coordCrop.row, coordCrop.col + MAX_DISPARITY);
    for(size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity) {
        TensorCoord coordRightImage(coordLeftImage.row, coordLeftImage.col - disparity);
        distances(coordCrop.row, coordCrop.col, disparity) = 
            calcDistance(leftCSCT.colPtr(coordLeftImage),
                         rightCSCT.colPtr(coordRightImage),
                         compPerPixel);
    }
}

Tensor<uint32_t> allocateDistancesArray(size_t width, size_t height)
{
    TensorDims distShape (height, width - MAX_DISPARITY, MAX_DISPARITY+1);
    return {distShape, true};
}

Tensor<uint32_t> hamming(Tensor<bool>& leftCSCT, Tensor<bool>& rightCSCT)
{
    size_t width = leftCSCT.dims.cols;
    size_t height = leftCSCT.dims.rows;

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(height) / BLOCK_SIZE));
    size_t blocksHorz = static_cast<size_t>(
        ceil((static_cast<float>(width) - MAX_DISPARITY) / BLOCK_SIZE));
    dim3 numBlocks(blocksHorz, blocksVert);

    Tensor<uint32_t> distancesDev = allocateDistancesArray(width, height);

    hammingKernel<<<numBlocks, threadsPerBlock>>>(leftCSCT, rightCSCT, distancesDev);
    cudaDeviceSynchronize();

    return distancesDev;
}