#include "csct.cuh"
#include "util/format.hpp"

__device__ size_t indexInImage(size_t width, size_t height)
{
    int xCoordInImg = blockDim.x * blockIdx.x + threadIdx.x - RADIUS;
    int yCoordInImg = blockDim.y * blockIdx.y + threadIdx.y - RADIUS;

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
    size_t diameter = 2 * RADIUS + 1;
    size_t resultSize = ((diameter * (diameter + 1)) / 2) - 2;

    size_t resIdx = 0;
    int radiusInt = static_cast<int>(RADIUS);
    for (int diffY = -radiusInt; diffY <= radiusInt; ++diffY)
    {
        for (int diffX = -radiusInt; diffX <= diffY; ++diffX)
        {
            if (diffY == 0 && diffX == 0)
                continue;
            else if (diffY == RADIUS && diffX == radiusInt)
                continue;
            size_t diffIdx = (threadIdx.y + diffY) * blockDim.x + (threadIdx.x + diffX);
            size_t oppIdx = (threadIdx.y - diffY) * blockDim.x + (threadIdx.x - diffX);

            results[(resultSize * imageIndex) + resIdx] = imageBlock[diffIdx] >= imageBlock[oppIdx];
            ++resIdx;
        }
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
    return ((diameter * (diameter + 1)) / 2) - 2;
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
