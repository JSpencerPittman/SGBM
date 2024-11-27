#include "directional.cuh"

#include <stdio.h>

struct ImgCoord {
    __host__ __device__ ImgCoord(int x, int y): x(x), y(y) {}
    int x, y;
};

namespace Direction {
    enum Direction {LeftToRight, TopLeftToBotRight, TopToBot, TopRightToBotLeft, RightToLeft};

    __device__ ImgCoord start(Direction direction, size_t idx, size_t width, size_t height) {
        switch (direction) {
            case LeftToRight:
                return {0, static_cast<int>(idx)};
            case TopLeftToBotRight:
                if (idx < height) return {0, static_cast<int>(idx)};
                else return {static_cast<int>(idx - height + 1), 0};
            case TopToBot:
                return {static_cast<int>(idx), 0};
            case TopRightToBotLeft:
                if (idx < width) return {static_cast<int>(idx), 0};
                else return {static_cast<int>(width-1), static_cast<int>(idx - width + 1)};
            case RightToLeft:
                return {static_cast<int>(width-1), static_cast<int>(idx)};
            default:
                return {0, 0};
        }
    }

    __device__ ImgCoord next(Direction direction, ImgCoord& curr) {
        switch (direction) {
            case LeftToRight:
                 return {curr.x+1, curr.y};
            case TopLeftToBotRight:
                return {curr.x+1, curr.y+1};
            case TopToBot:
                return {curr.x, curr.y+1};
            case TopRightToBotLeft:
                return {curr.x-1, curr.y+1};
            case RightToLeft:
                return {curr.x-1, curr.y};
            default:
                return {0, 0};
        }
    }

    __host__ __device__ size_t maxSpan(Direction direction, size_t width, size_t height) {
        switch (direction) {
            case LeftToRight:
                 return height;
            case TopLeftToBotRight:
                return height + width - 1;
             case TopToBot:
                return width;
            case TopRightToBotLeft:
                return height + width -1;
            case RightToLeft:
                return height;
            default:
                return 0;
        }
    }

    __device__ bool inImage(ImgCoord& curr, size_t width, size_t height) {
        if(curr.x < 0 || curr.x >= width) return false;
        else if(curr.y < 0 || curr.y >= height) return false;
        else return true;
    }
};

namespace DisparityArray {
    __device__ size_t pixelLocation(ImgCoord imgCoord, size_t width) {
        size_t pixelIndex = imgCoord.y * width + imgCoord.x;
        return pixelIndex * (MAX_DISPARITY + 1);
    }
};

__device__ float minLossAtPixel(float* pixelLoss) {
    float minLoss = pixelLoss[0];
    for(size_t disparity = 1; disparity <= MAX_DISPARITY; ++disparity)
        minLoss = min(minLoss, pixelLoss[disparity]);
    return minLoss;
}

__device__ void calcLossesAtPixel(float* prevPixLoss, float* pixLoss, uint32_t* pixDist) {
    float minLossAtPrevPixel = minLossAtPixel(prevPixLoss);

    for(size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity) {
        float smoothLoss = prevPixLoss[disparity];
        if(disparity > 0) smoothLoss = min(smoothLoss, prevPixLoss[disparity - 1] + P1);
        if(disparity < MAX_DISPARITY) smoothLoss = min(smoothLoss, prevPixLoss[disparity + 1] + P1);
        smoothLoss = min(smoothLoss, minLossAtPrevPixel + P2);
        pixLoss[disparity] = pixDist[disparity] + smoothLoss;
    }
}

__device__ void addDirLossToAggregateLoss(float* dirLoss, float* aggLoss) {
    for(size_t disparity = 0; disparity <= MAX_DISPARITY; ++disparity)
        aggLoss[disparity] += dirLoss[disparity];
} 

__global__ void directionalLossKernel(Direction::Direction direction, uint32_t* distances, float* aggLoss, float* dirLoss, size_t width, size_t height, size_t maxSpan) {
    size_t gridIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(gridIdx >= maxSpan) return;

    ImgCoord pixCoord = Direction::start(direction, threadIdx.x, width, height);
    // Locations within the disparity arrays
    size_t pixLoc = DisparityArray::pixelLocation(pixCoord, width);
    uint32_t* pixDist = distances + pixLoc;
    float* pixDirLoss = dirLoss + pixLoc;
    float* pixAggLoss = aggLoss + pixLoc;

    // Initial row is equal to the hamming distances
    for(size_t disparity = 0; disparity <= MAX_DISPARITY+1; ++disparity) {
        pixDirLoss[disparity] = pixDist[disparity];
    }
    addDirLossToAggregateLoss(pixDirLoss, pixAggLoss);

    float* prevPixDirLoss = pixDirLoss;
    pixCoord = Direction::next(direction, pixCoord);

    while(Direction::inImage(pixCoord, width, height)) {
        pixLoc = DisparityArray::pixelLocation(pixCoord, width);
        pixDist = distances + pixLoc;
        pixDirLoss = dirLoss + pixLoc;
        pixAggLoss = aggLoss + pixLoc;

        calcLossesAtPixel(prevPixDirLoss, pixDirLoss, pixDist);
        addDirLossToAggregateLoss(pixDirLoss, pixAggLoss);

        prevPixDirLoss = pixDirLoss;
        pixCoord = Direction::next(direction, pixCoord);
    }
}

__global__ void disparityMapKernel(uint8_t* dispMap, float* loss, size_t width, size_t height) {
    ImgCoord pixCoord(blockIdx.x * blockDim.x + threadIdx.x,
                      blockIdx.y * blockDim.y + threadIdx.y);

    if(pixCoord.x >= width || pixCoord.y >= height) return;

    size_t pixLocInLoss = DisparityArray::pixelLocation(pixCoord, width);    
    loss = loss +  pixLocInLoss;
    size_t minDisparity = 0;
    float minLoss = loss[0];
    for(size_t disparity = 1; disparity <= MAX_DISPARITY; ++disparity) {
        if(loss[disparity] < minLoss) {
            minLoss = loss[disparity];
            minDisparity = disparity;
        }
    }
    size_t imageIdx = pixCoord.y * static_cast<int>(width) + pixCoord.x;
    dispMap[imageIdx] = static_cast<uint8_t>(minDisparity);
}

float* allocateLoss(size_t width, size_t height) {
    size_t numBytes = width * height * (MAX_DISPARITY + 1) * sizeof(float);
    float* lossDev;
    cudaMalloc(&lossDev, numBytes);
    cudaMemset(lossDev, 0, numBytes);
    return lossDev;
}

uint8_t* allocateDisparityMap(size_t width, size_t height) {
    size_t numBytes = width * height * sizeof(uint8_t);
    uint8_t* dispMapDev;
    cudaMalloc(&dispMapDev, numBytes);
    return dispMapDev;
}

void clearLoss(float* lossDev, size_t width, size_t height) {
    size_t numBytes = width * height * (MAX_DISPARITY + 1) * sizeof(float);
    cudaMemset(lossDev, 0, numBytes);
}

uint32_t* copyDistancesToDev(HamDistances& distancesHost) {
    uint32_t* distancesDev;
    cudaMalloc(&distancesDev, distancesHost.numBytes);
    cudaMemcpy(distancesDev, distancesHost.data, distancesHost.numBytes, cudaMemcpyHostToDevice);
    return distancesDev;
}

void lossInDirection(Direction::Direction direction, uint32_t* distances, float* aggLoss, float* dirLoss, size_t width, size_t height) {
    size_t numThreads = BLOCK_SIZE * BLOCK_SIZE;
    dim3 threadsPerBlock(numThreads);

    size_t maxSpan = Direction::maxSpan(direction, width, height);
    dim3 numBlocks(ceil(static_cast<float>(maxSpan) / BLOCK_SIZE));

    directionalLossKernel<<<numBlocks, threadsPerBlock>>>(direction, distances, aggLoss, dirLoss, width, height, maxSpan);
}

uint8_t* copyDisparityArrayToHost(uint8_t* dispArrDev, size_t width, size_t height) {
    size_t numPixels = width * height;
    uint8_t* dispArrHost = new uint8_t[numPixels];
    cudaMemcpy(dispArrHost, dispArrDev, numPixels * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    return dispArrHost;
}


uint8_t* directionalLoss(HamDistances& distancesHost, size_t width, size_t height) {
    // distances: row x col x disparity
    uint32_t* distancesDev = copyDistancesToDev(distancesHost);
    // losses: row * col * disparity
    float* aggLossesDev = allocateLoss(width, height);
    float* dirLossesDev = allocateLoss(width, height);

    for (int direction = Direction::LeftToRight; direction <= Direction::RightToLeft; ++direction) {
        lossInDirection((Direction::Direction)direction, distancesDev, aggLossesDev, dirLossesDev, width, height);
        clearLoss(dirLossesDev, width, height);
    }

    cudaFree(distancesDev);
    cudaFree(dirLossesDev);

    uint8_t* dispMapDev = allocateDisparityMap(width, height);
    
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(height) / BLOCK_SIZE));
    size_t blocksHorz = static_cast<size_t>(
        ceil(static_cast<float>(width) / BLOCK_SIZE));
    dim3 numBlocks(blocksHorz, blocksVert);

    disparityMapKernel<<<numBlocks, threadsPerBlock>>>(dispMapDev, aggLossesDev, width, height);
    cudaDeviceSynchronize();
    cudaFree(aggLossesDev);

    uint8_t* dispArrHost = copyDisparityArrayToHost(dispMapDev, width, height);
    // Synchronize and check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("2: CUDA Kernel Launch Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    cudaFree(dispMapDev);

    return dispArrHost;
}