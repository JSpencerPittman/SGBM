#include "sgbm.h"

struct ImgCoord
{
    __host__ __device__ ImgCoord(int x, int y) : x(x), y(y) {}
    int x, y;
};

namespace Direction
{
    enum Direction
    {
        LeftToRight,
        TopLeftToBotRight,
        TopToBot,
        TopRightToBotLeft,
        RightToLeft
    };

    __device__ ImgCoord start(Direction direction, size_t idx, size_t width, size_t height)
    {
        switch (direction)
        {
        case LeftToRight:
            return {0, static_cast<int>(idx)};
        case TopLeftToBotRight:
            if (idx < height)
                return {0, static_cast<int>(idx)};
            else
                return {static_cast<int>(idx - height + 1), 0};
        case TopToBot:
            return {static_cast<int>(idx), 0};
        case TopRightToBotLeft:
            if (idx < width)
                return {static_cast<int>(idx), 0};
            else
                return {static_cast<int>(width - 1), static_cast<int>(idx - width + 1)};
        case RightToLeft:
            return {static_cast<int>(width - 1), static_cast<int>(idx)};
        default:
            return {0, 0};
        }
    }

    __device__ ImgCoord next(Direction direction, ImgCoord &curr)
    {
        switch (direction)
        {
        case LeftToRight:
            return {curr.x + 1, curr.y};
        case TopLeftToBotRight:
            return {curr.x + 1, curr.y + 1};
        case TopToBot:
            return {curr.x, curr.y + 1};
        case TopRightToBotLeft:
            return {curr.x - 1, curr.y + 1};
        case RightToLeft:
            return {curr.x - 1, curr.y};
        default:
            return {0, 0};
        }
    }

    __host__ __device__ size_t maxSpan(Direction direction, size_t width, size_t height)
    {
        switch (direction)
        {
        case LeftToRight:
            return height;
        case TopLeftToBotRight:
            return height + width - 1;
        case TopToBot:
            return width;
        case TopRightToBotLeft:
            return height + width - 1;
        case RightToLeft:
            return height;
        default:
            return 0;
        }
    }

    __device__ bool inImage(ImgCoord &curr, size_t width, size_t height)
    {
        if (curr.x < 0 || curr.x >= width)
            return false;
        else if (curr.y < 0 || curr.y >= height)
            return false;
        else
            return true;
    }
};

__device__ float minLossAtPixel(float *pixelLoss)
{
    float minLoss = pixelLoss[0];
    for (size_t disparity = 1; disparity <= MAX_DISPARITY; ++disparity)
        minLoss = min(minLoss, pixelLoss[disparity]);
    return minLoss;
}

__device__ void calcLossesAtPixel(float *prevPixLoss, float *pixLoss, uint32_t *pixDist)
{
    float minLossAtPrevPixel = minLossAtPixel(prevPixLoss);

    for (size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity)
    {
        float smoothLoss = prevPixLoss[disparity];
        if (disparity > 0)
            smoothLoss = min(smoothLoss, prevPixLoss[disparity - 1] + P1);
        if (disparity < MAX_DISPARITY)
            smoothLoss = min(smoothLoss, prevPixLoss[disparity + 1] + P1);
        smoothLoss = min(smoothLoss, minLossAtPrevPixel + P2);
        pixLoss[disparity] = pixDist[disparity] + smoothLoss;
    }
}

__device__ void addDirLossToAggregateLoss(float *dirLoss, float *aggLoss)
{
    for (size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity)
        aggLoss[disparity] += dirLoss[disparity];
}

__global__ void directionalLossKernel(Direction::Direction direction,
                                      Tensor<uint32_t> distances,
                                      Tensor<float> aggLoss,
                                      Tensor<float> dirLoss,
                                      size_t maxSpan)
{
    size_t width = distances.dims.cols;
    size_t height = distances.dims.rows;

    size_t gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (gridIdx >= maxSpan)
        return;

    bool toggle = false;
    float *prevDirLoss = dirLoss.colPtr(gridIdx, toggle);
    float *currDirLoss = dirLoss.colPtr(gridIdx, !toggle);

    ImgCoord pixCoord = Direction::start(direction, threadIdx.x, width, height);
    // Locations within the disparity arrays
    float *pixAggLoss = aggLoss.colPtr(pixCoord.y, pixCoord.x);

    // Initial row is equal to the hamming distances
    for (size_t disparity = 0; disparity <= MAX_DISPARITY + 1; ++disparity)
        prevDirLoss[disparity] = distances(pixCoord.y, pixCoord.x, disparity);
    addDirLossToAggregateLoss(prevDirLoss, pixAggLoss);

    pixCoord = Direction::next(direction, pixCoord);

    while (Direction::inImage(pixCoord, width, height))
    {
        pixAggLoss = aggLoss.colPtr(pixCoord.y, pixCoord.x);

        calcLossesAtPixel(prevDirLoss, currDirLoss, distances.colPtr(pixCoord.y, pixCoord.x));
        addDirLossToAggregateLoss(currDirLoss, pixAggLoss);

        toggle = !toggle;
        prevDirLoss = dirLoss.colPtr(gridIdx, toggle);
        currDirLoss = dirLoss.colPtr(gridIdx, !toggle);
        pixCoord = Direction::next(direction, pixCoord);
    }
}

__global__ void disparityMapKernel(Tensor<Byte> dispMap, Tensor<float> loss,
                                   size_t width, size_t height)
{
    ImgCoord pixCoord(blockIdx.x * blockDim.x + threadIdx.x,
                      blockIdx.y * blockDim.y + threadIdx.y);

    if (pixCoord.x >= width || pixCoord.y >= height)
        return;

    size_t minDisparity = 0;
    float minLoss = loss(pixCoord.y, pixCoord.x, 0);
    for (size_t disparity = 1; disparity <= MAX_DISPARITY; ++disparity)
    {
        if (loss(pixCoord.y, pixCoord.x, disparity) < minLoss)
        {
            minLoss = loss(pixCoord.y, pixCoord.x, disparity);
            minDisparity = disparity;
        }
    }

    dispMap(pixCoord.y, pixCoord.x) = static_cast<Byte>(minDisparity);
}

Tensor<float> allocateAggregateLoss(size_t width, size_t height)
{
    TensorDims lossDims(height, width, MAX_DISPARITY + 1);
    Tensor<float> lossDev(lossDims, true);
    cudaMemset(lossDev.data, 0, lossDev.bytes());
    return lossDev;
}

Tensor<float> allocateDirectionalLoss(size_t maxSpan)
{
    TensorDims lossDims(maxSpan, 2, MAX_DISPARITY + 1);
    Tensor<float> lossDev(lossDims, true);
    return lossDev;
}

Tensor<Byte> allocateDisparityMap(size_t width, size_t height)
{
    TensorDims dispMapDims(height, width, 1);
    Tensor<Byte> dispMapDev(dispMapDims, true);
    return dispMapDev;
}

void clearLoss(Tensor<float> &lossDev)
{
    cudaMemset(lossDev.data, 0, lossDev.bytes());
}

void lossInDirection(Direction::Direction direction, Tensor<uint32_t> &distances,
                     Tensor<float> &aggLoss)
{
    size_t numThreads = BLOCK_SIZE * BLOCK_SIZE;
    dim3 threadsPerBlock(numThreads);

    size_t maxSpan = Direction::maxSpan(direction, distances.dims.cols, distances.dims.rows);
    dim3 numBlocks(ceil(static_cast<float>(maxSpan) / BLOCK_SIZE));

    Tensor<float> dirLoss = allocateDirectionalLoss(maxSpan);

    directionalLossKernel<<<numBlocks, threadsPerBlock>>>(direction, distances, aggLoss, dirLoss, maxSpan);

    dirLoss.free();
}

Tensor<Byte> directionalLoss(Tensor<uint32_t> &distances)
{
    size_t width = distances.dims.cols;
    size_t height = distances.dims.rows;

    // distances: row x col x disparity
    // losses: row * col * disparity
    Tensor<float> aggLossesDev = allocateAggregateLoss(width, height);

    for (int direction = Direction::LeftToRight; direction <= Direction::RightToLeft; ++direction)
    {
        // lossInDirection((Direction::Direction)direction, distancesDev, aggLossesDev);
        lossInDirection((Direction::Direction)direction, distances, aggLossesDev);
    }

    Tensor<Byte> dispMapDev = allocateDisparityMap(width, height);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(height) / BLOCK_SIZE));
    size_t blocksHorz = static_cast<size_t>(
        ceil(static_cast<float>(width) / BLOCK_SIZE));
    dim3 numBlocks(blocksHorz, blocksVert);

    disparityMapKernel<<<numBlocks, threadsPerBlock>>>(dispMapDev, aggLossesDev, width, height);
    cudaDeviceSynchronize();
    aggLossesDev.free();

    cudaDeviceSynchronize();

    return dispMapDev;
}