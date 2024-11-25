#ifndef DIRECTIONAL_H_
#define DIRECTIONAL_H_

#include<utility>
#include<cuda_runtime.h>

#include "hamming.cuh"

#define P1 0.3
#define P2 0.4

uint8_t* directionalLoss(HamDistances& distancesHost, size_t width, size_t height);

#endif