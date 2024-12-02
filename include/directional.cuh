#ifndef DIRECTIONAL_H_
#define DIRECTIONAL_H_

#include "hamming.cuh"

Tensor<Byte> directionalLoss(Tensor<uint32_t>& distancesHost);

#endif