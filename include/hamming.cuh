#ifndef HAMMING_CUH_
#define HAMMING_CUH_

#include <cstdint>

#include "csct.cuh"

#define MAX_DISPARITY 300

typedef Tensor<uint32_t> Distances;

Distances hamming(CSCTResults& leftCSCT, CSCTResults& rightCSCT);

#endif