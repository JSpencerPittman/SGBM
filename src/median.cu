#include "median.cuh"

#include <algorithm>

__device__ TensorCoord coordInImageMedian(Tensor<Byte>& image)
{
    int tileSize = BLOCK_SIZE - (2 * MEDIAN_RADIUS );
    int xCoordInImg = (blockIdx.x * tileSize) - MEDIAN_RADIUS + threadIdx.x;
    int yCoordInImg = (blockIdx.y * tileSize) - MEDIAN_RADIUS + threadIdx.y;

    /*
    If a coordinate lays outside of the image's bounds then the closest pixel is used for that value.
    This allows us to have a radius around the selected tile even for pixels on the edges of the image.
    */
    int xCoordClampedInImg = min(max(0, xCoordInImg), (int)image.dims.cols - 1);
    int yCoordClampedInImg = min(max(0, yCoordInImg), (int)image.dims.rows - 1);

    return {static_cast<size_t>(yCoordClampedInImg), static_cast<size_t>(xCoordClampedInImg)};
}

__device__ bool insideHaloMedian()
{
    if (threadIdx.x < MEDIAN_RADIUS || blockDim.x - MEDIAN_RADIUS <= threadIdx.x)
        return false;
    else if (threadIdx.y < MEDIAN_RADIUS || blockDim.y - MEDIAN_RADIUS <= threadIdx.y)
        return false;
    return true;
}

__device__ void selectionSort(Byte* arr, size_t len) {
    for(size_t idxInsert = 0; idxInsert < len-1; ++idxInsert) {
        size_t minIdx = idxInsert;
        for(size_t idxSearch = idxInsert+1; idxSearch < len; ++idxSearch)
            if(arr[idxSearch] < arr[minIdx]) minIdx = idxSearch;

        Byte temp = arr[idxInsert];
        arr[idxInsert] = arr[minIdx];
        arr[minIdx] = temp;
    }
}


__device__ void medianBlurAtPixel(Tensor<Byte>& dispMapBlock,
                                  Tensor<Byte>& blurDispMap,
                                  TensorCoord coordDispMap,
                                  Byte* kernelCache) {
    for(int diffY = -MEDIAN_RADIUS; diffY <= MEDIAN_RADIUS; ++diffY)
        for(int diffX = -MEDIAN_RADIUS; diffX <= MEDIAN_RADIUS; ++diffX)
            kernelCache[(diffY + MEDIAN_RADIUS) * MEDIAN_DIAMETER + (diffX + MEDIAN_RADIUS)] = 
                dispMapBlock(threadIdx.y + diffY, threadIdx.x + diffX);

    selectionSort(kernelCache, MEDIAN_DIAMETER * MEDIAN_DIAMETER);
    size_t middleIdx = ((MEDIAN_DIAMETER * MEDIAN_DIAMETER) - 1) / 2;

    Byte median = kernelCache[middleIdx];
    blurDispMap(coordDispMap) = median;
}

__global__ void medianBlurKernel(Tensor<Byte> dispMap, Tensor<Byte> blurDispMap) {
    __shared__ Byte dispMapBlockData[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ Byte kernelCache[MEDIAN_TILE_SIZE * MEDIAN_TILE_SIZE * MEDIAN_DIAMETER * MEDIAN_DIAMETER];

    Tensor<Byte> dispMapBlock({BLOCK_SIZE, BLOCK_SIZE, 1}, (Byte*)&dispMapBlockData);

    TensorCoord coordDispMap = coordInImageMedian(dispMap);
    dispMapBlock(threadIdx.y, threadIdx.x) = dispMap(coordDispMap);
    __syncthreads();

    if (insideHaloMedian()) {
        size_t tileIdx = (threadIdx.y - MEDIAN_RADIUS) * MEDIAN_TILE_SIZE + (threadIdx.x - MEDIAN_RADIUS);
        medianBlurAtPixel(dispMapBlock, blurDispMap, coordDispMap, (Byte*)(kernelCache + tileIdx * (MEDIAN_DIAMETER * MEDIAN_DIAMETER)));
    }
}

Tensor<Byte> allocateBlurDispMap(Tensor<Byte>& image) {
    return {image.dims, true};
}

Tensor<Byte> medianBlur(Tensor<Byte>& dispMap) {
    Tensor<Byte> blurDispMap = allocateBlurDispMap(dispMap);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    float tileSize = static_cast<float>(BLOCK_SIZE) - (2 * static_cast<float>(MEDIAN_RADIUS));
    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(dispMap.dims.rows) / tileSize));
    size_t blocksHorz = static_cast<size_t>(
        ceil(static_cast<float>(dispMap.dims.cols) / tileSize));
    dim3 numBlocks(blocksHorz, blocksVert);

    medianBlurKernel<<<numBlocks, threadsPerBlock>>>(dispMap, blurDispMap);
    // Synchronize and check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("2: CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    Tensor<Byte> blurDispMapHost = blurDispMap.copyToHost();

    blurDispMap.free();

    return blurDispMapHost;
}
