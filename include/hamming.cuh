#ifndef HAMMING_CUH_
#define HAMMING_CUH_

#include <cstdint>

#include "csct.cuh"

Tensor<uint32_t> hamming(Tensor<bool>& leftCSCT, Tensor<bool>& rightCSCT);

#endif