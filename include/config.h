#ifndef CONFIG_H_
#define CONFIG_H_

// Center-Symmetric Census Transform
#define BLOCK_SIZE 32
#define CSCT_RADIUS 2
#define CSCT_DIAMETER RADIUS*2 + 1

// Hamming Distances
#define MAX_DISPARITY 64

// Directional Loss
#define P1 7
#define P2 17
#define NUM_DIRECTIONS 5

// Median Blur
#define MEDIAN_RADIUS 2
#define MEDIAN_DIAMETER (MEDIAN_RADIUS*2 + 1)
#define MEDIAN_TILE_SIZE (BLOCK_SIZE - (MEDIAN_RADIUS*2))

// Dilation Erosion
#define MORPH_ITERATIONS 2
#define MORPH_RADIUS 3
#define MORPH_DIAMETER (MORPH_RADIUS*2 + 1)
#define MORPH_TILE_SIZE (BLOCK_SIZE - (MORPH_RADIUS*2))


#endif