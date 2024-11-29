#ifndef MEDIAN_CUH_
#define MEDIAN_CUH_

#include "csct.cuh"

#define MEDIAN_RADIUS 2
#define MEDIAN_DIAMETER (MEDIAN_RADIUS*2 + 1)
#define MEDIAN_TILE_SIZE (BLOCK_SIZE - (MEDIAN_RADIUS*2))

FlatImage medianBlur(FlatImage& dispMap);

#endif