#include "sgbm.cuh"

__device__ TensorCoord coordInImageRankFilt(Tensor<Byte> &image)
{
    int tileSize = BLOCK_SIZE - (2 * MORPH_RADIUS);
    int xCoordInImg = (blockIdx.x * tileSize) - MORPH_RADIUS + threadIdx.x;
    int yCoordInImg = (blockIdx.y * tileSize) - MORPH_RADIUS + threadIdx.y;

    /*
    If a coordinate lays outside of the image's bounds then the closest pixel is used for that value.
    This allows us to have a radius around the selected tile even for pixels on the edges of the image.
    */
    int xCoordClampedInImg = min(max(0, xCoordInImg), (int)image.dims.cols - 1);
    int yCoordClampedInImg = min(max(0, yCoordInImg), (int)image.dims.rows - 1);

    return {static_cast<size_t>(yCoordClampedInImg), static_cast<size_t>(xCoordClampedInImg)};
}

__device__ bool insideHaloRankFilt()
{
    if (threadIdx.x < MORPH_RADIUS || blockDim.x - MORPH_RADIUS <= threadIdx.x)
        return false;
    else if (threadIdx.y < MORPH_RADIUS || blockDim.y - MORPH_RADIUS <= threadIdx.y)
        return false;
    return true;
}

__device__ void selectionSort(Byte *arr, size_t len)
{
    for (size_t idxInsert = 0; idxInsert < len - 1; ++idxInsert)
    {
        size_t minIdx = idxInsert;
        for (size_t idxSearch = idxInsert + 1; idxSearch < len; ++idxSearch)
            if (arr[idxSearch] < arr[minIdx])
                minIdx = idxSearch;

        Byte temp = arr[idxInsert];
        arr[idxInsert] = arr[minIdx];
        arr[minIdx] = temp;
    }
}

__device__ void medianRankFilterAtPixel(Tensor<Byte> &dispMapBlock,
                                        Tensor<Byte> &blurDispMap,
                                        TensorCoord coordDispMap,
                                        Byte *kernelCache)
{
    for (int diffY = -MORPH_RADIUS; diffY <= MORPH_RADIUS; ++diffY)
        for (int diffX = -MORPH_RADIUS; diffX <= MORPH_RADIUS; ++diffX)
            kernelCache[(diffY + MORPH_RADIUS) * MORPH_DIAMETER + (diffX + MORPH_RADIUS)] =
                dispMapBlock(threadIdx.y + diffY, threadIdx.x + diffX);

    selectionSort(kernelCache, MORPH_DIAMETER * MORPH_DIAMETER);
    size_t middleIdx = ((MORPH_DIAMETER * MORPH_DIAMETER) - 1) / 2;

    Byte median = kernelCache[middleIdx];
    blurDispMap(coordDispMap) = median;
}

__global__ void medianRankFilterKernel(Tensor<Byte> dispMap, Tensor<Byte> blurDispMap)
{
    __shared__ Byte dispMapBlockData[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ Byte kernelCache[MORPH_TILE_SIZE * MORPH_TILE_SIZE * MORPH_DIAMETER * MORPH_DIAMETER];

    Tensor<Byte> dispMapBlock({BLOCK_SIZE, BLOCK_SIZE, 1}, (Byte *)&dispMapBlockData);

    TensorCoord coordDispMap = coordInImageRankFilt(dispMap);
    dispMapBlock(threadIdx.y, threadIdx.x) = dispMap(coordDispMap);
    __syncthreads();

    if (insideHaloRankFilt())
    {
        size_t tileIdx = (threadIdx.y - MORPH_RADIUS) * MORPH_TILE_SIZE + (threadIdx.x - MORPH_RADIUS);
        Byte *kernelCacheAtPixel = kernelCache + tileIdx * (MORPH_DIAMETER * MORPH_DIAMETER);
        medianRankFilterAtPixel(dispMapBlock, blurDispMap, coordDispMap, kernelCacheAtPixel);
    }
}

__device__ void minMaxFilterAtPixel(Tensor<Byte> &dispMapBlock,
                                    Tensor<Byte> &blurDispMap,
                                    TensorCoord coordDispMap,
                                    bool isMaxOperation)
{
    // bool log = blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 2 && threadIdx.y == 2;
    // if(log) printf("Latched onto target thread.\n");
    // if(log) printf("%lu, %lu, %lu\n", dispMapBlock.dims.rows, dispMapBlock.dims.cols, dispMapBlock.dims.channels);
    Byte value = blurDispMap(coordDispMap);
    for (int diffY = -MORPH_RADIUS; diffY <= MORPH_RADIUS; ++diffY)
        for (int diffX = -MORPH_RADIUS; diffX <= MORPH_RADIUS; ++diffX)
        {
            if (isMaxOperation)
                value = max(value, dispMapBlock(threadIdx.y + diffY, threadIdx.x + diffX));
            else
                value = min(value, dispMapBlock(threadIdx.y + diffY, threadIdx.x + diffX));
        }

    blurDispMap(coordDispMap) = value;
}

__global__ void minMaxRankFilterKernel(Tensor<Byte> dispMap, Tensor<Byte> blurDispMap,
                                       bool isMaxOperation)
{
    __shared__ Byte dispMapBlockData[BLOCK_SIZE * BLOCK_SIZE];

    Tensor<Byte> dispMapBlock({BLOCK_SIZE, BLOCK_SIZE, 1}, (Byte *)&dispMapBlockData);

    TensorCoord coordDispMap = coordInImageRankFilt(dispMap);
    dispMapBlock(threadIdx.y, threadIdx.x) = dispMap(coordDispMap);
    __syncthreads();

    if (insideHaloRankFilt())
        minMaxFilterAtPixel(dispMapBlock, blurDispMap, coordDispMap, isMaxOperation);
}

Tensor<Byte> allocateBlurDispMap(Tensor<Byte> &image)
{
    return {image.dims, true};
}

Tensor<Byte> rankFilter(Tensor<Byte> &dispMap, RankOperation operation)
{
    Tensor<Byte> blurDispMap = allocateBlurDispMap(dispMap);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    float tileSize = static_cast<float>(BLOCK_SIZE) - (2 * static_cast<float>(MORPH_RADIUS));
    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(dispMap.dims.rows) / tileSize));
    size_t blocksHorz = static_cast<size_t>(
        ceil(static_cast<float>(dispMap.dims.cols) / tileSize));
    dim3 numBlocks(blocksHorz, blocksVert);

    if (operation == RankOperation::Median)
        medianRankFilterKernel<<<numBlocks, threadsPerBlock>>>(dispMap, blurDispMap);
    else
        minMaxRankFilterKernel<<<numBlocks, threadsPerBlock>>>(dispMap, blurDispMap, operation == RankOperation::Maximum);
    // Synchronize and check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("2: CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return blurDispMap;
}
