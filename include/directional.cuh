#ifndef DIRECTIONAL_H_
#define DIRECTIONAL_H_

#include<utility>
#include<cuda_runtime.h>

#include "hamming.cuh"

#define P1 0.3
#define P2 0.4

struct ImgCoord {
    ImgCoord(int x, int y): x(x), y(y) {}
    int x, y;
};

namespace Direction {
    __device__ ImgCoord start(size_t idx, size_t width, size_t height) {
        return {0, idx};
    }

    __device__ ImgCoord prev(ImgCoord& curr) {
        return {curr.x+1, curr.y};
    }

    __device__ ImgCoord next(ImgCoord& curr) {
        return {curr.x+1, curr.y};
    }

    __device__ size_t maxSpan(size_t width, size_t height) {
        return height;
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

uint8_t* directionalLoss(HamDistances& distancesHost, size_t width, size_t height);

#endif