#include "csct.cuh"

__device__ TensorCoord coordInImage(Tensor<Byte>& image)
{
    int tileSize = BLOCK_SIZE - (2 * RADIUS );
    int xCoordInImg = (blockIdx.x * tileSize) - RADIUS + threadIdx.x;
    int yCoordInImg = (blockIdx.y * tileSize) - RADIUS + threadIdx.y;

    /*
    If a coordinate lays outside of the image's bounds then the closest pixel is used for that value.
    This allows us to have a radius around the selected tile even for pixels on the edges of the image.
    */
    int xCoordClampedInImg = min(max(0, xCoordInImg), (int)image.dims.cols - 1);
    int yCoordClampedInImg = min(max(0, yCoordInImg), (int)image.dims.rows - 1);

    return {static_cast<size_t>(yCoordClampedInImg), static_cast<size_t>(xCoordClampedInImg)};
}

__device__ bool insideHalo()
{
    if (threadIdx.x < RADIUS || blockDim.x - RADIUS <= threadIdx.x)
        return false;
    else if (threadIdx.y < RADIUS || blockDim.y - RADIUS <= threadIdx.y)
        return false;
    return true;
}

__device__ void censusTransform(Tensor<Byte>& imageBlock, Tensor<bool>& csctResults, TensorCoord coordImage)
{
    size_t compIdx = 0;
    int radiusInt = static_cast<int>(RADIUS);
    for(int diffX = -radiusInt; diffX < 0; ++diffX) {
        for(int diffY = -radiusInt; diffY <= radiusInt; ++diffY) {
            csctResults(coordImage.row, coordImage.col, compIdx) = 
                imageBlock(threadIdx.y + diffY, threadIdx.x + diffX) >=
                imageBlock(threadIdx.y - diffY, threadIdx.x - diffX);
            ++compIdx;
        }
    }
    for(int diffY = -radiusInt; diffY < 0; ++diffY) {
        csctResults(coordImage.row, coordImage.col, compIdx) = 
                imageBlock(threadIdx.y + diffY, threadIdx.x) >=
                imageBlock(threadIdx.y - diffY, threadIdx.x);
        ++compIdx;
    }
}

__global__ void csctKernel(Tensor<Byte> image, Tensor<bool> csctResults)
{
    __shared__ Byte imageBlockData[BLOCK_SIZE * BLOCK_SIZE];
    Tensor<Byte> imageBlock({BLOCK_SIZE, BLOCK_SIZE, 1}, (Byte*)&imageBlockData);

    TensorCoord coordImage = coordInImage(image);
    imageBlock(threadIdx.y, threadIdx.x) = image(coordImage);
    __syncthreads();

    if (insideHalo())
        censusTransform(imageBlock, csctResults, coordImage);
};

Tensor<bool> allocateCSCTResultArray(Image& image) {
    size_t diameter = 2 * RADIUS + 1;
    size_t compPerPix = diameter * RADIUS + RADIUS;
    TensorDims csctResShape(image.height(), image.width(), compPerPix);
    Tensor<bool> resultsDev(csctResShape, true);
    return resultsDev;
}

Tensor<bool> csct(Image &image)
{
    // Load images onto GPU
    Tensor<Byte> imageDev = image.data()->copyToDevice();
    Tensor<bool> resultsDev = allocateCSCTResultArray(image);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    float tileSize = static_cast<float>(BLOCK_SIZE) - (2 * static_cast<float>(RADIUS));
    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(image.height()) / tileSize));
    size_t blocksHorz = static_cast<size_t>(
        ceil(static_cast<float>(image.width()) / tileSize));
    dim3 numBlocks(blocksHorz, blocksVert);

    csctKernel<<<numBlocks, threadsPerBlock>>>(imageDev, resultsDev);
    cudaDeviceSynchronize();

    imageDev.free();

    return resultsDev;
}
