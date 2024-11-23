#include <cmath>
#include <iostream>

#include "sgbm.cuh"
#include "csct.cuh"

// __global__ void testKernel(Byte* image, bool* intensities, size_t width, size_t height) {
//     size_t xCoordInImg = blockDim.x * blockIdx.x + threadIdx.x;
//     size_t yCoordInImg = blockDim.y * blockIdx.y + threadIdx.y;

//     if(xCoordInImg < width && yCoordInImg < height) {
//         size_t imageIndex = yCoordInImg * width + xCoordInImg;
//         intensities[imageIndex] = image[imageIndex] > 127;
//     }
// }

// void test(Image& image) {
//     Byte* imageDev;
//     cudaMalloc(&imageDev, image.size());
//     cudaMemcpy(imageDev, image.data(), image.size(), cudaMemcpyHostToDevice);

//     bool* resultsDev;
//     size_t resultSize = image.width() * image.height();
//     cudaMalloc(&resultsDev, resultSize * sizeof(bool));

//     dim3 threadsPerBlock (BLOCK_SIZE, BLOCK_SIZE);

//     size_t blocksVert = static_cast<size_t>(
//         ceil(static_cast<float>(image.height()) / BLOCK_SIZE));
//     size_t blocksHorz = static_cast<size_t>(
//         ceil(static_cast<float>(image.width()) / BLOCK_SIZE));
//     dim3 numBlocks(blocksHorz, blocksVert);

//     testKernel<<<numBlocks, threadsPerBlock>>>(imageDev, resultsDev, image.width(), image.height());

//     bool* resultsHost = new bool[resultSize];
//     cudaMemcpy(resultsHost, resultsDev, resultSize * sizeof(bool), cudaMemcpyDeviceToHost);

//     for(size_t idx = 0; idx < image.width(); ++idx)
//         std::cout << (int)resultsHost[idx];
//     std::cout << std::endl;

//     cudaFree(imageDev);
//     cudaFree(resultsDev);

//     delete [] resultsHost;
// }

__global__ void sgbmKernel(Byte* image, bool* csctResults, size_t width, size_t height) {
    csct(image, csctResults, width, height, RADIUS);
}

void sgbm(Image& image) {
    Byte* imageDev;
    cudaMalloc(&imageDev, image.size());
    cudaMemcpy(imageDev, image.data(), image.size(), cudaMemcpyHostToDevice);
    
    bool* csctResDev;
    size_t numResults = image.width() * image.height();
    size_t diameter = 2 * RADIUS + 1;
    size_t resultSize = ((diameter * (diameter + 1)) / 2) - 2;
    size_t csctResSize = resultSize * numResults;
    cudaMalloc(&csctResDev, csctResSize * sizeof(bool));

    dim3 threadsPerBlock (BLOCK_SIZE, BLOCK_SIZE);

    float tileSize = static_cast<float>(BLOCK_SIZE) - (2 * static_cast<float>(RADIUS));
    size_t blocksVert = static_cast<size_t>(
        ceil(static_cast<float>(image.height()) / tileSize));
    size_t blocksHorz = static_cast<size_t>(
        ceil(static_cast<float>(image.width()) / tileSize));
    dim3 numBlocks(blocksHorz, blocksVert);

    sgbmKernel<<<numBlocks, threadsPerBlock>>>(imageDev, csctResDev, image.width(), image.height());
        
    bool* csctResHost = new bool[csctResSize];
    cudaMemcpy(csctResHost, csctResDev, csctResSize * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(imageDev);
    cudaFree(csctResDev);

    std::cout << "IMAGESIZE:" << image.width() << " x " << image.height() << std::endl;
    std::cout << "RESULTSIZE:" << resultSize << std::endl;

    for(size_t idx = 0; idx < image.width(); ++idx) {
        for(size_t idx2 = 0; idx2 < resultSize; ++idx2)
            std::cout << static_cast<int>(csctResHost[idx*resultSize + idx2]);
        std::cout << " ";
    }
    std::cout << std::endl;

    delete [] csctResHost;

    printf("It has been done.\n");
}