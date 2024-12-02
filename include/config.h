#ifndef CONFIG_H_
#define CONFIG_H_

// Center-Symmetric Census Transform
#define BLOCK_SIZE 32
#define RADIUS 3

// Hamming Distances
#define MAX_DISPARITY 300

// Directional Loss
#define P1 7
#define P2 17

// Median
#define MEDIAN_RADIUS 2
#define MEDIAN_DIAMETER (MEDIAN_RADIUS*2 + 1)
#define MEDIAN_TILE_SIZE (BLOCK_SIZE - (MEDIAN_RADIUS*2))

#endif