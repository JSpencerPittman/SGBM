#ifndef HAMMING_CUH_
#define HAMMING_CUH_

#include <cstdint>

#include "csct.cuh"

#define MAX_DISPARITY 200

typedef Tensor<uint32_t> Distances;

// struct HamDistances
// {
//     HamDistances(uint32_t *data, size_t numPixels, size_t maxDisparity) : data(data), numPixels(numPixels), maxDisparity(maxDisparity), numBytes(numPixels * (maxDisparity+1) * sizeof(uint32_t)) {}

//     uint32_t *pixel(size_t pixelIdx)
//     {
//         return data + maxDisparity * pixelIdx;
//     }

//     uint32_t &distance(size_t pixelIdx, size_t disparity)
//     {
//         return pixel(pixelIdx)[disparity];
//     }

//     uint32_t *data;
//     size_t numPixels;
//     size_t maxDisparity;
//     size_t numBytes;
// };

Distances hamming(CSCTResults& leftCSCT, CSCTResults& rightCSCT);

#endif