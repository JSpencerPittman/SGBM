#include "csct.cuh"
#include "util/format.hpp"

__device__ size_t indexInImage(size_t width, size_t height)
{
    size_t tileSize = BLOCK_SIZE - (2 * RADIUS );
    int xCoordInImg = (blockIdx.x * tileSize) - RADIUS + threadIdx.x;
    int yCoordInImg = (blockIdx.y * tileSize) - RADIUS + threadIdx.y;

    /*
    If a coordinate lays outside of the image's bounds then the closest pixel is used for that value.
    This allows us to have a radius around the selected tile even for pixels on the edges of the image.
    */
    size_t xCoordClampedInImg = min(static_cast<size_t>(max(0, xCoordInImg)), width - 1);
    size_t yCoordClampedInImg = min(static_cast<size_t>(max(0, yCoordInImg)), height - 1);

    return yCoordClampedInImg * width + xCoordClampedInImg;
}

__device__ void saveImageBlockToSharedMemory(Byte *image, Byte imageBlock[], size_t imageIndex)
{
    size_t blockIdx = blockDim.x * threadIdx.y + threadIdx.x;
    imageBlock[blockIdx] = image[imageIndex];
}

__device__ bool insideHalo()
{
    if (threadIdx.x < RADIUS || blockDim.x - RADIUS <= threadIdx.x)
        return false;
    else if (threadIdx.y < RADIUS || blockDim.y - RADIUS <= threadIdx.y)
        return false;
    return true;
}

__device__ void censusTransform(Byte *imageBlock, bool *results, size_t imageIndex)
{
    //temp
    size_t xIdx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yIdx = blockIdx.y * blockDim.y + threadIdx.y;
    bool log = blockIdx.x == 2 && blockIdx.y == 3 && threadIdx.x == 13 && threadIdx.y == 17;
    if (log) {printf("XIDX: %lu, YIDX: %lu\n", xIdx, yIdx);}

    size_t diameter = 2 * RADIUS + 1;
    size_t compPerPixel = diameter * RADIUS + RADIUS;

    size_t compIdx = 0;
    int radiusInt = static_cast<int>(RADIUS);
    if(log) printf("CENSUS-INSIDE for %d\n", threadIdx.y * blockDim.x + threadIdx.x);
    for(int diffX = -radiusInt; diffX < 0; ++diffX) {
        for(int diffY = -radiusInt; diffY <= radiusInt; ++diffY) {
            int diffIdx = (threadIdx.y + diffY) * blockDim.x + (threadIdx.x + diffX);
            int oppIdx = (threadIdx.y - diffY) *  blockDim.x + (threadIdx.x - diffX);

            if(log) {
                printf("Diff: (%d, %d) Comparing %d to %d, IDXS: %d and %d\n", diffX, diffY, imageBlock[diffIdx], imageBlock[oppIdx], diffIdx, oppIdx);
            }

            results[(compPerPixel * imageIndex) + compIdx] = imageBlock[diffIdx] >= imageBlock[oppIdx];
            ++compIdx;
        }
    }
    for(int diffY = -radiusInt; diffY < 0; ++diffY) {
        int diffIdx = (threadIdx.y + diffY) * blockDim.x + threadIdx.x;
        int oppIdx = (threadIdx.y - diffY) * blockDim.x + threadIdx.x;

        results[(compPerPixel * imageIndex) + compIdx] = imageBlock[diffIdx] >= imageBlock[oppIdx];
        ++compIdx;
    }
}

__global__ void csctKernel(Byte *image, bool *results, size_t width, size_t height)
{
    __shared__ Byte imageBlock[BLOCK_SIZE * BLOCK_SIZE];

    size_t imageIndex = indexInImage(width, height);
    saveImageBlockToSharedMemory(image, imageBlock, imageIndex);
    __syncthreads();

    if (insideHalo())
        censusTransform(imageBlock, results, imageIndex);
};

size_t comparisonsPerPixel(size_t width, size_t height)
{
    size_t diameter = 2 * RADIUS + 1;
    return diameter * RADIUS + RADIUS;
}

CSCTResults allocateCSCTResultArray(size_t width, size_t height)
{
    bool *csctResDev;
    size_t numPixels = width * height;
    size_t compPerPix = comparisonsPerPixel(width, height);
    size_t totalComparisons = compPerPix * numPixels;
    size_t numBytes = totalComparisons * sizeof(bool);
    cudaMalloc(&csctResDev, numBytes);
    return {csctResDev, numPixels, compPerPix};
}

Byte *copyImageToDevice(Image &image)
{
    Byte *imageDev;
    cudaMalloc(&imageDev, image.size());
    cudaMemcpy(imageDev, image.data(), image.size(), cudaMemcpyHostToDevice);
    return imageDev;
}

CSCTResults copyResultsToHost(CSCTResults resultsDev)
{
    CSCTResults resultsHost(new bool[resultsDev.numBytes],
                            resultsDev.numPixels,
                            resultsDev.compPerPix);
    cudaMemcpy(resultsHost.data, resultsDev.data, resultsDev.numBytes, cudaMemcpyDeviceToHost);
    return resultsHost;
}

CSCTResults csct(Image &image)
{
    size_t width = image.width(), height = image.height(), imageSize = image.size();

    // Load images onto GPU
    Byte *imageDev = copyImageToDevice(image);
    CSCTResults resultsDev = allocateCSCTResultArray(image.width(), image.height());

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    float tileSize = static_cast<float>(BLOCK_SIZE) - (2 * static_cast<float>(RADIUS));
    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(height) / tileSize));
    size_t blocksHorz = static_cast<size_t>(
        ceil(static_cast<float>(width) / tileSize));
    dim3 numBlocks(blocksHorz, blocksVert);

    csctKernel<<<numBlocks, threadsPerBlock>>>(imageDev, resultsDev.data, width, height);

    CSCTResults resultsHost = copyResultsToHost(resultsDev);

    cudaFree(imageDev);
    cudaFree(resultsDev.data);

    return resultsHost;
}
